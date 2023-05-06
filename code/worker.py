from ast import With
from typing_extensions import Annotated
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, Subset, Dataset
import os
from models import *
import time
import warnings

from models.two_layer_net import TwoLayerNet, SynthNetwork
warnings.filterwarnings("ignore")

def compute_norm(parameters):
    with torch.no_grad():
        total_norm = 0
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        for p in parameters:
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

    return total_norm


class WorkerLtd:
    def __init__(
        self, 
        w_ind, 
        network, 
        train_subset, 
        validation_subset, 
        logdir, 
        use_sigmoid='relu',    # last layer activation fnction
        test_subset=None, 
        test_subset_full=None,
        discr_subset=None,
        num_classes=10,
        num_workers=20,
        lr=0.01,
        weight_decay=5e-4,
        lr_scheduler='none',
        train_batch_size=256,
        hetero_arch=False, 
        embedding_dim=84,
        arch_allocation=0,
        transfer='none',
        num_moments=1,
        weights=None, 
        lamda=1.0,
        lamda_l2=1.0,
        lr_gamma=1.0,
        mean_computation_dataset='train_subset',
        mean_computation_batch_size=256,
        normalize=False,
        per_layer_lr=False,
        device='cpu',
        h_dim=3,
        out_dim=2,
        writer=None,
        reload=False,
    ) -> None:

        self.worker_ind = w_ind
        self.logdir = os.path.join(logdir,'checkpoints', str(w_ind))
        if not reload:
            os.makedirs(self.logdir)
        self.device = device

        if 'synth' in network:
            self.model = SynthNetwork(h_dim=h_dim,out_dim=out_dim, num_classes=num_classes)
            self.embedding_dim = out_dim
        elif 'twolayer-lenet-hetero' in network:
            # assert hetero_arch is False
            if arch_allocation==0:
                self.model = LeNetBounded()
            elif arch_allocation==1:
                self.model = LeNetBounded1()
            elif arch_allocation==2:
                self.model = TwoLayerNet(hidden=512, latent=84)
            elif arch_allocation==3:
                self.model = TwoLayerNet(hidden=256, latent=84)
            else:
                raise NotImplementedError
            self.embedding_dim = 84

        elif 'twolayer' in network:
            self.model = TwoLayerNet()
            self.embedding_dim = 100
        elif 'lenet' in network:
            if hetero_arch:
                if arch_allocation==0:
                    self.model = LeNetBounded(embedding_dim)
                elif arch_allocation==1:
                    self.model = LeNetBounded1(embedding_dim)
                else:
                    raise NotImplementedError
            else:
                self.model = LeNetBounded(embedding_dim)
            self.embedding_dim = embedding_dim
        else:
            raise NotImplementedError

        self.model = self.model.to(self.device)
        self.num_classes = num_classes

        self.iter = 0
        self.transfer = transfer
        self.weights=None

        self.reg_loss = torch.nn.MSELoss(reduction='mean')
        self.weights = torch.FloatTensor(weights).to(self.device)

        # Loss function, optimizer, learning rate, batch size etc
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr=lr
        self.weight_decay = weight_decay

        if 'synth' in network:
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=self.lr,
            )        
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=self.lr,
                momentum=0.9, 
                weight_decay=self.weight_decay
            )
            
        self.lr_scheduler = lr_scheduler
        self.lr_gamma = lr_gamma
        self.scheduler = None

        self.lamda = lamda
        self.lamda_l2 = lamda_l2
        self.normalize = normalize

        self.train_batch_size = train_batch_size
        self.mean_computation_batch_size = mean_computation_batch_size
        self.mean_computation_dataset = mean_computation_dataset

        # load datasets and data loaders
        self.train_subset = train_subset
        self.validation_subset = validation_subset
        self.test_subset = test_subset
        if 'synth' in network:
            self.train_loader = DataLoader(train_subset, batch_size=self.train_batch_size, shuffle=True)
            self.validation_loader = DataLoader(validation_subset, batch_size=512, shuffle=True)
            self.test_loader = DataLoader(self.test_subset, batch_size=512, shuffle=False)
        else:
            # self.train_loader = DataLoader(train_subset, batch_size=self.train_batch_size, shuffle=True, num_workers=8, pin_memory=True)
            # self.validation_loader = DataLoader(validation_subset, batch_size=512, shuffle=True, num_workers=8, pin_memory=True)
            if test_subset_full is not None:
                self.test_subset_full = test_subset_full
                self.test_loader_full = DataLoader(self.test_subset_full, batch_size=512, shuffle=False,num_workers=8, pin_memory=True)
            self.train_loader = DataLoader(train_subset, batch_size=self.train_batch_size, shuffle=True, pin_memory=True)
            self.validation_loader = DataLoader(validation_subset, batch_size=512, shuffle=True, pin_memory=True)
            self.test_loader = DataLoader(self.test_subset, batch_size=512, shuffle=False,num_workers=8, pin_memory=True)

        self.discr_subset = discr_subset
        if discr_subset is None:
            self.discr_loader = None
        else:
            self.discr_loader = DataLoader(self.discr_subset, batch_size=512, shuffle=False)

        self.best_acc = 0

        self.train_acc = 0
        self.test_acc = 0
        self.valid_acc = 0
        self.test_acc_full = 0

        self.predictions = None
        self.num_workers = num_workers
        self.mean_embeddings = torch.zeros(self.num_workers, self.num_classes, self.embedding_dim)

        self.num_moments = num_moments
        self.moments = torch.zeros(self.num_workers, self.num_classes, self.embedding_dim, num_moments)
        self.marginal_moments = torch.zeros(self.num_workers, self.embedding_dim, num_moments)

        self.cons_embeddings = torch.zeros(self.num_classes, self.embedding_dim, num_moments)
        self.cons_marginal_embeddings = torch.zeros(self.embedding_dim, num_moments)

        self.global_classifier = nn.Linear(self.embedding_dim, self.num_classes).to(self.device)
        self.server_model = LeNetBounded(embedding_dim)

        self.writer = writer

    def train(self, epoch, use_transfer=True):
        tt=time.time()
        self.model.train()
        correct = 0
        total = 0

        train_loss = 0
        train_loss_classification = 0

        regularization_losses = {
            'total_loss': 0,
            'moment_1_loss': 0,
            'l2_reg': 0,
        }
        losses = {
            'total_loss': 0,
            'class_loss': 0
        }
        gradient_norms = {
            'features': 0,
            'classifier': 0,
            'total': 0
        }
        param_norms = {
            'features': 0,
            'classifier': 0,
            'total': 0
        }

        time_taken = {
            'forward_pass': 0,
            'label_embeddings': 0,
            'moment_1_loss': 0,
        }
        train_time = 0
        log_time=0
        epoch_time=0
        ttt=time.time()
        model_load_time=ttt-tt

        for moment_ in range(2,self.num_moments+1):
            time_taken[f'moment_{moment_}_loss'] = 0
            regularization_losses[f'moment_{moment_}_loss'] = 0
        regularization_losses['total_moment_loss'] = 0
        
        batch_idx = -1
        for inputs, targets in self.train_loader:
            batch_idx += 1

            t0=time.time()
            if batch_idx==0:
                loader_time=t0-ttt

            self.iter += 1
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            embeddings, outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)

            train_loss_classification += loss.item()

            if self.transfer == 'fedcda':

                temp_loss = torch.tensor(0.0, requires_grad=True)
                if not (self.cons_embeddings == 0).all() and use_transfer:
                    self.cons_embeddings = self.cons_embeddings.to(self.device)

                    for label in range(self.num_classes):
                        if sum(targets==label) > 0:

                            label_embeddings = embeddings[(targets==label)]

                            temp_loss = self.reg_loss(label_embeddings, self.cons_embeddings[label,:,0])
                            regularization_losses[f'moment_1_loss'] += temp_loss.item()

                            # Compute CMD Loss
                            centralized = label_embeddings - label_embeddings.mean(axis=0)
                            for moment_ in range(1, self.num_moments):
                                centralized_powk = centralized**(moment_+1)
                                mean_centralized_powk = centralized_powk.mean(axis=0)

                                temp_moment_loss = self.reg_loss(mean_centralized_powk, self.cons_embeddings[label,:,moment_])
                                regularization_losses[f'moment_{moment_+1}_loss'] += temp_moment_loss.item()
                                temp_loss = temp_loss + temp_moment_loss

                            loss = loss + temp_loss * self.lamda

                # Computation of L2 loss ||w-\bar{w}||^2
                l2_loss = torch.tensor(0.0, requires_grad=True)
                for local_param, global_param in zip(self.model.classifier.parameters(), self.global_classifier.parameters()):
                    l2_loss = l2_loss + torch.norm(local_param-global_param, 2)**2

                regularization_losses['l2_reg'] += l2_loss.item()
                regularization_losses['total_moment_loss'] += temp_loss.item()
                regularization_losses['total_loss'] += self.lamda * temp_loss.item() + self.lamda_l2 * l2_loss.item()

                loss = loss + l2_loss * self.lamda_l2
                if torch.isnan(loss):
                    print(label, embeddings.shape, embeddings[(targets==label)].shape, embeddings[(targets==label)].mean(axis=0).norm())
                    raise 

            elif self.transfer == 'fedcda-marginal':
                temp_loss = torch.tensor(0.0, requires_grad=True)
                if not (self.cons_marginal_embeddings == 0).all() and use_transfer:
                    self.cons_marginal_embeddings = self.cons_marginal_embeddings.to(self.device)
                    temp_loss = self.reg_loss(embeddings, self.cons_marginal_embeddings[:,0])
                    regularization_losses[f'moment_1_loss'] += temp_loss.item()

                    centralized = embeddings - embeddings.mean(axis=0)
                    for moment_ in range(1, self.num_moments):
                        centralized_powk = centralized**(moment_+1)
                        mean_centralized_powk = centralized_powk.mean(axis=0)
                        temp_moment_loss = self.reg_loss(mean_centralized_powk, self.cons_marginal_embeddings[:,moment_])
                        regularization_losses[f'moment_{moment_+1}_loss'] += temp_moment_loss.item()
                        temp_loss = temp_loss + temp_moment_loss

                    loss = loss + temp_loss * self.lamda

                # Computation of L2 loss ||w-\bar{w}||^2
                l2_loss = torch.tensor(0.0, requires_grad=True)
                for local_param, global_param in zip(self.model.classifier.parameters(), self.global_classifier.parameters()):
                    l2_loss = l2_loss + torch.norm(local_param-global_param, 2)**2

                regularization_losses['l2_reg'] += l2_loss.item()
                regularization_losses['total_moment_loss'] += temp_loss.item()
                regularization_losses['total_loss'] += self.lamda * temp_loss.item() + self.lamda_l2 * l2_loss.item()

                loss = loss + l2_loss * self.lamda_l2
                if torch.isnan(loss):
                    print(label, embeddings.shape, embeddings[(targets==label)].shape, embeddings[(targets==label)].mean(axis=0).norm())
                    raise 

            loss.backward()
            self.optimizer.step()
            t1=time.time()
            train_time += t1-t0

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            with torch.no_grad():
                feature_param_grads = [param.grad.data.cpu() for name, param in self.model.named_parameters() if 'classifier' not in name]
                classifier_param_grads = [param.grad.data.cpu() for name, param in self.model.named_parameters() if 'classifier' in name]
                total_param_grads = [param.grad.data.cpu() for name, param in self.model.named_parameters()]

                gradient_norms['features'] += compute_norm(feature_param_grads)
                gradient_norms['classifier'] += compute_norm(classifier_param_grads)
                gradient_norms['total'] += compute_norm(total_param_grads)

                param_norms['features'] += compute_norm([param.data.cpu() for name, param in self.model.named_parameters() if 'classifier' not in name])
                param_norms['classifier'] += compute_norm([param.data.cpu() for name, param in self.model.named_parameters() if 'classifier' in name])
                param_norms['total'] += compute_norm([param.data.cpu() for name, param in self.model.named_parameters()])

                losses['total_loss'] = train_loss/(batch_idx+1)
                losses['class_loss'] = train_loss_classification/(batch_idx+1)
            
            t2=time.time()
            log_time+= t2-t1
        epoch_time=time.time()-ttt


        self.train_acc = 100.*correct/total
        return losses, regularization_losses, time_taken, batch_idx+1, self.optimizer.param_groups[0]['lr'], gradient_norms, param_norms

    def inference(self, epoch, valid=False, verbose=True, full_test=False):

        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        if valid:
            loader = self.validation_loader
        else:
            if full_test:
                loader = self.test_loader_full
            else:
                loader = self.test_loader

        with torch.no_grad():
            batch_idx = -1
            for inputs, targets in loader:
                batch_idx += 1
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                _, outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                # if verbose:
                #     progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #                 % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if valid:
            self.writer.add_scalar(f'Valid Acc/Worker_{self.worker_ind}', 100.*correct/total, self.iter)
            self.writer.add_scalar(f'Valid Loss/Worker_{self.worker_ind}', test_loss/(batch_idx+1), self.iter)
            self.writer.add_scalar(f'Valid Acc epoch/Worker_{self.worker_ind}', 100.*correct/total, epoch)
            self.writer.add_scalar(f'Valid Loss epoch/Worker_{self.worker_ind}', test_loss/(batch_idx+1), epoch)
            # Save checkpoint.
            self.valid_acc = 100.*correct/total
            if self.valid_acc >= self.best_acc:
                # print('Saving..')
                state = {
                    'net': self.model.state_dict(),
                    'acc': self.valid_acc,
                    'epoch': epoch,
                }
                torch.save(state, os.path.join(self.logdir, 'ckpt.pth'))
                self.best_acc = self.valid_acc

        elif full_test:
            self.writer.add_scalar(f'FullTest Acc/Worker_{self.worker_ind}', 100.*correct/total, self.iter)
            self.writer.add_scalar(f'FullTest Loss/Worker_{self.worker_ind}', test_loss/(batch_idx+1), self.iter)
            self.writer.add_scalar(f'FullTest Acc epoch/Worker_{self.worker_ind}', 100.*correct/total, epoch)
            self.writer.add_scalar(f'FullTest Loss epoch/Worker_{self.worker_ind}', test_loss/(batch_idx+1), epoch)
            self.test_acc_full = 100.*correct/total
        else:
            self.writer.add_scalar(f'Test Acc/Worker_{self.worker_ind}', 100.*correct/total, self.iter)
            self.writer.add_scalar(f'Test Loss/Worker_{self.worker_ind}', test_loss/(batch_idx+1), self.iter)
            self.writer.add_scalar(f'Test Acc epoch/Worker_{self.worker_ind}', 100.*correct/total, epoch)
            self.writer.add_scalar(f'Test Loss epoch/Worker_{self.worker_ind}', test_loss/(batch_idx+1), epoch)
            self.test_acc = 100.*correct/total

    def compute_forward_pass(self, shuffle=False, discr=False):
        if discr and (self.discr_loader is not None):
            loader = self.discr_loader
        else:
            if 'train' in self.mean_computation_dataset:
                temp_dataset = self.train_subset
            else:
                temp_dataset = self.validation_subset

            loader = DataLoader(
                temp_dataset, batch_size=self.mean_computation_batch_size, shuffle=shuffle)

        self.model.eval()
        with torch.no_grad():
            if shuffle:
                # use all of the dataset to compute the moment embeddings
                all_embeddings = []
                all_labels = []
                for batch, labels in loader:
                    batch, labels = batch.to(self.device), labels.to(self.device)
                    embeddings, _ = self.model(batch)
                    all_embeddings.append(embeddings)
                    all_labels.append(labels)
                embeddings=torch.cat(all_embeddings)
                labels=torch.cat(all_labels)

            else:
                for batch, labels in loader:
                    batch, labels = batch.to(self.device), labels.to(self.device)
                    embeddings, _ = self.model(batch)
                    break

        return embeddings, labels

    def compute_moment_embeddings(self, epoch, save_embeddings=True, logdir=None, shuffle=False):

        per_label_embeddings = torch.zeros(self.num_classes,self.embedding_dim, self.num_moments).to(self.device)
        embeddings, labels = self.compute_forward_pass(shuffle=shuffle)

        if save_embeddings:

            state = {
                'embeddings': embeddings.cpu(),
                'labels': labels.cpu(),
                'epoch': epoch,
            }
            if logdir is None:
                raise NotImplementedError

            if not os.path.exists(os.path.join(logdir, 'saved_embeddings')):
                os.makedirs(os.path.join(logdir, 'saved_embeddings'))
            torch.save(state, os.path.join(logdir, 'saved_embeddings', f'{self.worker_ind}-{epoch}.pth'))

        with torch.no_grad():
            for label in range(self.num_classes):
                if not (labels==label).any():
                    per_label_embeddings[label,:,:] = torch.zeros(embeddings.shape[1],self.num_moments)
                else:
                    per_label_embeddings[label,:,0] = embeddings[(labels==label)].mean(axis=0)

                    for moment_ in range(2, self.num_moments+1):
                        centralized = embeddings[(labels==label)] - per_label_embeddings[label,:,0]
                        centralized_powk = centralized**moment_
                        mean_centralized_powk = centralized_powk.mean(axis=0)

                        per_label_embeddings[label,:,moment_-1] = mean_centralized_powk

        # Store self mean embeddings
        # self.moments[self.worker_ind] = per_label_embeddings

        return per_label_embeddings

    def compute_marginal_moment_embeddings(self, epoch, save_embeddings=True, logdir=None, shuffle=False):

        marginal_embeddings = torch.zeros(self.embedding_dim, self.num_moments).to(self.device)
        embeddings, labels = self.compute_forward_pass(shuffle=shuffle)

        if save_embeddings:

            state = {
                'embeddings': embeddings.cpu(),
                'labels': labels.cpu(),
                'epoch': epoch,
            }
            if logdir is None:
                raise NotImplementedError

            if not os.path.exists(os.path.join(logdir, 'saved_embeddings')):
                os.makedirs(os.path.join(logdir, 'saved_embeddings'))
            torch.save(state, os.path.join(logdir, 'saved_embeddings', f'{self.worker_ind}-{epoch}.pth'))

        with torch.no_grad():
            marginal_embeddings[:,0] = embeddings.mean(axis=0)

            for moment_ in range(2, self.num_moments+1):
                centralized = embeddings - marginal_embeddings[:,0]
                centralized_powk = centralized**moment_
                mean_centralized_powk = centralized_powk.mean(axis=0)

                marginal_embeddings[:,moment_-1] = mean_centralized_powk

        # Store self mean embeddings
        self.marginal_moments[self.worker_ind] = marginal_embeddings

        return marginal_embeddings

    def store_cons_moment_embeddings(self, cons_embeddings):
        self.cons_embeddings = cons_embeddings.detach().clone()
        assert self.cons_embeddings.requires_grad is False

    def store_cons_marginal_moment_embeddings(self, marginal_embeddings):
        self.cons_marginal_embeddings = marginal_embeddings.detach().clone()
        assert self.cons_marginal_embeddings.requires_grad is False

    def store_global_classifier(self, classifier):
        with torch.no_grad():
            for p1, p2 in zip(self.global_classifier.parameters(), classifier.parameters()):
                p1.data = p2.detach().clone().data
                p1.requires_grad=False

        self.global_classifier.eval()

    def store_global_model(self, server_model):
        with torch.no_grad():
            for p1, p2 in zip(self.server_model.parameters(), server_model.parameters()):
                p1.data = p2.detach().clone().data
                p1.requires_grad=False

        self.server_model.eval()
    