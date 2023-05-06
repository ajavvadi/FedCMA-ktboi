from typing_extensions import Annotated
from sklearn.feature_extraction import img_to_graph
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import glob
from PIL import Image

from torch.utils.data import DataLoader, Sampler, Subset, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import shutil
import argparse
import random
from models import *
from worker import WorkerLtd, compute_norm
from data_utils import get_transforms, get_datasets, load_saved_embeddings
from compute_divergence import train_discriminator, create_dataset_from_saved_embeddings, SingleLayerNet
from plot_utils import plot_moments, plot_space

import time
import datetime
import json
import csv
# from csv import writer
import pickle
import wandb
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)


def experiment_name(args):
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    if args.num_workers > 6:
        ratios = 'large'
    else:
        ratios='-'.join(list(args.ratios.split(',')))

    if args.share_last_layer and args.weighted_sharing:
        share_last_layer = f'{args.share_last_layer}-{args.weighted_sharing}'
    else:
        share_last_layer = f'{args.share_last_layer}'
    share_last_layer = f'{share_last_layer}-{args.use_weight_transfer_after}'

    if args.active_ratio < 1.0:
        num_workers = f'{args.num_workers}-{args.active_ratio}'
    else:
        num_workers = f'{args.num_workers}'

    if 'fedcda' in args.transfer:
        transfer = f'{args.transfer}-{args.lamda}-{args.lamda_l2}-{args.mean_computation_dataset}-{args.mean_computation_batch_size}-{share_last_layer}-{args.num_moments}'
    else:
        transfer = f'{args.transfer}'

    lr = f'{args.lr}-{args.weight_decay}-{args.lr_scheduler}-{args.lr_gamma}-{args.lr_alpha}-{args.lr_decay}-{args.lrl2_gamma}-{args.lrl2_alpha}-{args.lrl2_decay}'

    name = f'{time_stamp}_{ratios}_{args.net}-{args.h_dim}-{args.out_dim}_{num_workers}_{args.seed}_{lr}-{args.batch_size}_{transfer}'

    if 'synth' in args.data_path:
        name = f'{name}_synth'
    elif args.dirichlet:
        name=f'{name}_dirichlet-{args.alpha}'
    
    name=f'{name}_{args.use_sigmoid}'

    return name


def anneal_cons_alpha(args, epoch, l2=False,discr=1.0):
    if l2:
        val = args.lrl2_alpha * args.lamda_l2
        if args.lrl2_decay == 'none':
            pass
        elif args.lrl2_decay == 't':
            val = val/(1 + args.lrl2_gamma * epoch**(1))
        elif args.lrl2_decay == 'custom':
            val = val * (1-discr)
        else:
            raise NotImplementedError
    else:
        val = args.lr_alpha * args.lamda
        if args.lr_decay == 'none':
            pass
        elif args.lr_decay == 't':
            val = val/(1 + args.lr_gamma * epoch**(1))
        else:
            raise NotImplementedError

    return val


def initialize_dataset():

    return_val = torch.load(os.path.join(args.data_path, 'synthetic_dataset.pt'))
    num_classes=4

    return return_val, num_classes


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

    parser.add_argument('--gitlog', default=' ')
    parser.add_argument('--net', default='synth', choices=['synth','twolayer','lenet'])
    parser.add_argument('--h_dim', default=3,type=int)
    parser.add_argument('--out_dim', default=2,type=int)

    parser.add_argument('--use_sigmoid', default='relu', choices=['relu','sigmoid'])

    parser.add_argument('--data_path', default='~/data/CIFAR10')
    parser.add_argument('--logdir', default='./logs')
    parser.add_argument('--group_name', default='group')
    parser.add_argument('--project', default='project')

    parser.add_argument('--data_seed', default=0, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--ratios', default='0.5,0.5')
    parser.add_argument('--dirichlet', default=False, action='store_true')
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--no_expert', default=False,action='store_true')   # Don't use last worker (with more data) in distillation
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_scheduler', default='none', help='learning rate scheduler')
    parser.add_argument('--lr_gamma', default=1.0, type=float, help='learning rate scheduler')
    parser.add_argument('--lr_alpha', default=0.9, type=float)
    parser.add_argument('--lr_decay', default='none')    
    parser.add_argument('--lrl2_gamma', default=1.0, type=float, help='learning rate scheduler')
    parser.add_argument('--lrl2_alpha', default=0.9, type=float)
    parser.add_argument('--lrl2_decay', default='none')    
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--per_layer_lr', default=False, action='store_true')

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lamda', default=0.5, type=float, help='regularization')
    parser.add_argument('--lamda_l2', default=0.5, type=float, help='regularization')

    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--use_transfer_until', default=50, type=int)
    parser.add_argument('--use_weight_transfer_after', default=0, type=int)


    parser.add_argument('--normalize',action="store_true", default=False)
    parser.add_argument('--mean_computation_dataset', default='train-subset')
    parser.add_argument('--mean_computation_batch_size', default=128, type=int)
    parser.add_argument('--shuffle', default='False')
    parser.add_argument('--share_last_layer', default=False, action='store_true')
    parser.add_argument('--weighted_sharing', default=False, action='store_true')

    parser.add_argument('--hetero_arch', default=False, action='store_true')
    parser.add_argument('--active_ratio', default=1.0, type=float)

    parser.add_argument('--transfer', default='none', choices=['none','fedcda'])
    parser.add_argument('--num_moments', default=1, type=int)
    parser.add_argument('--compute_divergence', default=False, action='store_true')
    parser.add_argument('--discr_comp_interval', default=50, type=int)

    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    ########################################################
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # LOGGING
    exp_name_ = experiment_name(args)
    args.logdir = os.path.join(args.logdir, exp_name_)
    os.makedirs(args.logdir)
    writer = SummaryWriter(args.logdir)
    embeddings_dir = os.path.join(args.logdir, 'embeddings')
    fig_dir = os.path.join(args.logdir, 'figures')
    os.makedirs(fig_dir)
    os.makedirs(embeddings_dir)
    wandb.init(
        project=args.project, 
        entity='jrr', 
        settings=wandb.Settings(start_method="fork"),
        name=exp_name_,
        config=args,
        group=args.group_name
    )

    # DATASETS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    loaded_dataset, num_classes = initialize_dataset()

    with open(os.path.join(args.logdir,'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    workers = {}
    for worker_ind in range(args.num_workers):
        discr_set = None
        if args.dirichlet:
            temp_test_set = loaded_dataset['test_subsets'][worker_ind]
        else:
            if 'synthetic' in args.data_path:
                temp_test_set = loaded_dataset['test_dataset']
                if 'discr_dataset' not in loaded_dataset:
                    discr_set = loaded_dataset['test_subsets'][worker_ind]
                else:
                    discr_set = loaded_dataset['discr_dataset'][worker_ind]
            elif 'synth-v2' in args.data_path:
                temp_test_set = loaded_dataset['test_dataset']
            elif 'synth-v1' in args.data_path:
                temp_test_set = loaded_dataset['test_subsets'][worker_ind]
            else:
                temp_test_set = loaded_dataset['test_subset']

        workers[worker_ind] = WorkerLtd(
            worker_ind, 
            args.net, 
            loaded_dataset['train_subsets'][worker_ind], 
            loaded_dataset['validation_subsets'][worker_ind], 
            args.logdir, 
            h_dim=args.h_dim,
            out_dim=args.out_dim,
            use_sigmoid=args.use_sigmoid,
            test_subset=temp_test_set,
            discr_subset=discr_set,
            num_classes=num_classes,
            transfer=args.transfer,
            num_moments=args.num_moments,
            weights=loaded_dataset['weights'],
            num_workers=args.num_workers,
            lamda=args.lamda,
            lamda_l2=args.lamda_l2,
            lr=args.lr,
            weight_decay=args.weight_decay,
            train_batch_size=args.batch_size,
            mean_computation_batch_size=args.mean_computation_batch_size,
            mean_computation_dataset=args.mean_computation_dataset,
            device=device,
            writer=writer
        )

    with open(os.path.join(args.logdir, f'combined_results.csv'), 'a+') as f_:
        csv_writer = csv.writer(f_,delimiter=',')
        csv_writer.writerow(['Best','Latest','Validation'])

    running_best_acc = 0
    running_best_accs = {}
    latest_accs = {}
    for _, worker in workers.items():

        running_best_accs[worker.worker_ind] = running_best_acc
        latest_accs[worker.worker_ind] = running_best_acc

    temp_embed_dim_ = args.out_dim
        
    consensus_embeddings = torch.zeros(num_classes,temp_embed_dim_,args.num_moments)
    weights = torch.FloatTensor(loaded_dataset['weights']).to(device)
    print(weights)
    assert weights.shape == (args.num_workers, num_classes)

    consensus_classifier = nn.Linear(temp_embed_dim_, num_classes).to(device)
    temp_consensus_classifier = nn.Linear(temp_embed_dim_, num_classes).to(device)

    # Synchronize last layer weights before starting training
    # Without this training might be slow and does not yield good performance
    with torch.no_grad():
        for _, worker in workers.items():
            for p1, p2 in zip(consensus_classifier.parameters(), worker.model.classifier.parameters()):
                p2.data = p1.detach().clone().data
            worker.store_global_classifier(consensus_classifier)

    discr_accuracy_total = 1.0
    train_time = 0
    log_time = 0
    infer_time = 0
    transfer_time = 0
    
    for epoch in tqdm(range(args.num_epochs)):
        ep_train_time = 0
        ep_log_time = 0
        ep_infer_time = 0
        if epoch <= args.use_transfer_until:
            use_transfer = True
        else:
            use_transfer = False

        row_wandb = {}
        for _, worker in workers.items():

            t0=time.time()
            losses, regularization_losses, time_taken, batch_idx, learning_rate, gradient_norms, param_norms = worker.train(epoch, use_transfer=use_transfer)
            t1=time.time()
            ep_train_time = ep_train_time + t1-t0            

            worker.inference(epoch, valid=True)
            worker.inference(epoch, valid=False)
            t2=time.time()
            ep_infer_time = ep_infer_time + t2-t1

            latest_accs[worker.worker_ind] = worker.test_acc
            if worker.valid_acc == worker.best_acc:
                running_best_acc = worker.test_acc
                running_best_accs[worker.worker_ind] = running_best_acc

            row_wandb[f'best_test_acc_{worker.worker_ind}'] = running_best_accs[worker.worker_ind],
            row_wandb[f'train_acc_{worker.worker_ind}'] = worker.train_acc,
            row_wandb[f'test_acc_{worker.worker_ind}'] = worker.test_acc,
            row_wandb[f'valid_acc_{worker.worker_ind}'] = worker.valid_acc,
            row_wandb[f'Learning_rate_{worker.worker_ind}'] = learning_rate
            row_wandb[f'Classification_loss_{worker.worker_ind}'] = losses['class_loss']
            row_wandb[f'Total_loss_{worker.worker_ind}'] = losses['total_loss']

            # average param grad per batch in the epoch
            row_wandb[f'Feature_grad_{worker.worker_ind}'] = gradient_norms['features']/batch_idx
            row_wandb[f'Classifier_grad_{worker.worker_ind}'] = gradient_norms['classifier']/batch_idx
            row_wandb[f'Total_grad_{worker.worker_ind}'] = gradient_norms['total']/batch_idx

            row_wandb[f'Feature_paramnorm_{worker.worker_ind}'] = param_norms['features']/batch_idx
            row_wandb[f'Classifier_paramnorm_{worker.worker_ind}'] = param_norms['classifier']/batch_idx
            row_wandb[f'Total_paramnorm_{worker.worker_ind}'] = param_norms['total']/batch_idx

            global_local_classifier = [p1 - p2 for (p1,p2) in zip(worker.model.classifier.parameters(),consensus_classifier.parameters())]
            row_wandb[f'Global-local_classifiernorm_{worker.worker_ind}'] = compute_norm(global_local_classifier)

            row_wandb[f'Reg_{worker.worker_ind}'] = regularization_losses[f'total_loss']/batch_idx
            row_wandb[f'Reg-L2_{worker.worker_ind}'] = regularization_losses[f'l2_reg']/batch_idx
            for moment_ in range(1,args.num_moments+1):
                row_wandb[f'Reg_M{moment_}_{worker.worker_ind}'] = regularization_losses[f'moment_{moment_}_loss']/batch_idx

            t3=time.time()
            ep_log_time = ep_log_time + t3-t2
        
        row_wandb['train_time'] = ep_train_time
        row_wandb['log_time'] = ep_log_time
        train_time += ep_train_time
        log_time += ep_log_time
        infer_time += ep_infer_time

        mean_accuracy = 0
        accuracies = [running_best_accs[worker_ind_] for worker_ind_ in running_best_accs.keys()]
        latest_accuracies = [latest_accs[worker_ind_] for worker_ind_ in latest_accs.keys()]
        valid_accuracies = [workers[_ind_].valid_acc for _ind_ in range(args.num_workers)]

        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        mean_latest_accuracy = np.mean(latest_accuracies)
        std_latest_accuracy = np.std(latest_accuracies)
        mean_valid_accuracy = np.mean(valid_accuracies)
        std_valid_accuracy = np.std(valid_accuracies)

        sum_accuracies={
            'mean_best_accuracy': mean_accuracy,
            'std_best_accuracy': std_accuracy,
            'mean_latest_accuracy': mean_latest_accuracy,
            'std_latest_accuracy': std_latest_accuracy,
            'mean_valid_accuracy': mean_valid_accuracy,
            'std_valid_accuracy': std_valid_accuracy
        }
        combined_res_row = [str(mean_accuracy), str(mean_latest_accuracy), str(mean_valid_accuracy)]
        with open(os.path.join(args.logdir, f'combined_results.csv'), 'a+') as f_:
            csv_writer = csv.writer(f_,delimiter=',')
            csv_writer.writerow(combined_res_row)

        ########## Communication ############

        # Transfer knowledge here
        t0=time.time()
        with torch.no_grad():
            moment_embeddings = torch.zeros(args.num_workers, num_classes, worker.embedding_dim, args.num_moments)
            for _, worker in workers.items():
                if args.shuffle=='True':
                    temp_embed = worker.compute_moment_embeddings(epoch=epoch, logdir=args.logdir, save_embeddings=False, shuffle=True)
                else:
                    temp_embed = worker.compute_moment_embeddings(epoch=epoch, logdir=args.logdir, save_embeddings=False, shuffle=False)

                for temp_lab in range(num_classes):
                    # Update after checking if the embeddings are not zeros (initialized values)
                    if not (temp_embed[temp_lab,:,:] == 0).all():
                        moment_embeddings[worker.worker_ind,temp_lab,:,:] = temp_embed[temp_lab,:,:]

            if torch.isnan(moment_embeddings).any():
                print('moment embeddings')
                raise

        assert not (moment_embeddings == 0).all()
        # This is not needed for transfer=none, but just computing it since it doesn't really get used in that case. 
        with torch.no_grad():

            moment_embeddings=moment_embeddings.to(device)
            consensus_embeddings=consensus_embeddings.to(device)

            row_wandb['norm(e1-e2)'] = torch.norm(moment_embeddings[0]- moment_embeddings[1]).item()
            row_wandb['norm(g-e1)'] = torch.norm(moment_embeddings[0]- consensus_embeddings).item()
            row_wandb['norm(g-e2)'] = torch.norm(moment_embeddings[1]- consensus_embeddings).item()

            alpha_t = anneal_cons_alpha(args, epoch, l2=False)
            cons_update_norm = 0
            for _label_ in range(num_classes):
                _cons_emb_=torch.einsum('b, bij -> ij', weights[:,_label_], moment_embeddings[:,_label_,:,:])
                cons_update_norm += torch.norm(consensus_embeddings[_label_,:,:] - _cons_emb_)**2
                consensus_embeddings[_label_,:,:] = (1-alpha_t) * consensus_embeddings[_label_,:,:] + alpha_t * _cons_emb_
                # find norm of _cons_emb_ for each label and plot the combined norm of the update. 
            cons_update_norm = torch.sqrt(cons_update_norm)
            row_wandb['consensus_update_norm'] = cons_update_norm

            for _, worker in workers.items():
                worker.store_cons_moment_embeddings(consensus_embeddings)
            
            np.save(os.path.join(embeddings_dir, f'epoch-{epoch}.npy'), np.array(consensus_embeddings.cpu()))

        with torch.no_grad():
            for p1, p2 in zip(temp_consensus_classifier.parameters(), workers[0].model.classifier.parameters()):
                p1.data = p2.detach().clone().data
            for p1, p2 in zip(temp_consensus_classifier.parameters(), workers[1].model.classifier.parameters()):
                p1.data = p1.data - p2.detach().clone().data
            
            row_wandb['norm(w1-w2)'] = compute_norm(temp_consensus_classifier.parameters())

        if (args.transfer != 'none') and args.share_last_layer and (epoch >= args.use_weight_transfer_after):

                for p1, p2 in zip(temp_consensus_classifier.parameters(), workers[0].model.classifier.parameters()):
                    if args.weighted_sharing:
                        p1.data = p2.detach().clone().data * (weights.sum(axis=1)[0]/weights.sum())
                        # p1.data = p2.data * weights[0].sum()
                    else:
                        p1.data = p2.detach().clone().data/args.num_workers

                for wid in range(1, args.num_workers):
                    worker = workers[wid]
                    for p1, p2 in zip(temp_consensus_classifier.parameters(), worker.model.classifier.parameters()):
                        if args.weighted_sharing:
                            p1.data = p1.data + (weights.sum(axis=1)[wid]/weights.sum()) * p2.detach().clone().data
                            # p1.data = p1.data + weights[wid].sum() * p2.data
                        else:
                            p1.data = p1.data + p2.detach().clone().data/args.num_workers

                alpha_t = anneal_cons_alpha(args, epoch, l2=True, discr=discr_accuracy_total)
                if 'fedcda' in args.transfer:
                    for p1, p2 in zip(consensus_classifier.parameters(), temp_consensus_classifier.parameters()):
                        p1.data = p1.data* (1-alpha_t) + alpha_t* p2.detach().clone().data

                    for wid, worker in workers.items():
                        worker.store_global_classifier(consensus_classifier)
        
        ###################################################
        # Compute divergence

        if ((epoch % args.discr_comp_interval==0) or (epoch==args.num_epochs-1)) and args.compute_divergence:
            for _, worker in workers.items():
                # discr_data, discr_emb, discr_pred, discr_lab = worker.compute_forward_pass_all(worker.discr_loader)
                discr_emb, discr_lab = worker.compute_forward_pass(shuffle=True, discr=True)

                state = {
                    # 'data': discr_data,
                    'embeddings': discr_emb.cpu(),
                    # 'predictions': discr_pred,
                    'labels': discr_lab.cpu(),
                    'epoch': epoch,
                }
                if not os.path.exists(os.path.join(args.logdir, 'saved_embeddings')):
                    os.makedirs(os.path.join(args.logdir, 'saved_embeddings'))
                torch.save(state, os.path.join(args.logdir, 'saved_embeddings', f'{worker.worker_ind}-{epoch}.pth'))

            discr = {}
            loaded_dataset = create_dataset_from_saved_embeddings(
                load_saved_embeddings(os.path.join(args.logdir, 'saved_embeddings'), epoch=epoch, workers=list(range(args.num_workers))),
                labels=list(range(num_classes)),
                workers=list(range(args.num_workers))
            )

            discr_batch_size = 128

            loader = DataLoader(
                loaded_dataset, batch_size=discr_batch_size, shuffle=True,
                pin_memory=True
            )
            model = SingleLayerNet(num_workers=args.num_workers, hsize=temp_embed_dim_).to(device)
            discrim_accuracy, discrim_divergence = train_discriminator(model, loader, device=device, num_epochs=500, lr=0.1,writer=writer, data_epoch=epoch)
            discr[f'discr_accuracy_total'] = discrim_accuracy
            discr[f'discr_divergence_total'] = discrim_divergence

            t1=time.time()
            row_wandb['transfer_time'] = t1-t0
            transfer_time = transfer_time + t1-t0
            wandb.log({**row_wandb, **sum_accuracies, **discr}, step=epoch)
            t2=time.time()

            # Delete the saved_embeddings epoch files. 
            # shutil.rmtree(os.path.join(args.logdir,'saved_embeddings'))

            with torch.no_grad():

                plot_moments(
                    global_moments=consensus_embeddings[:,:,0].detach().clone().cpu(),
                    local_moments=moment_embeddings[:,:,:,0].detach().clone().cpu(),
                    logdir=fig_dir,
                    epoch=epoch
                )
                for __w_id__ in range(args.num_workers):
                    plot_space(
                        __w_id__, 
                        workers[__w_id__].model, 
                        epoch,
                        workers[0].train_loader,
                        workers[1].train_loader,
                        workers[2].train_loader,
                        workers[3].train_loader,
                        logdir=fig_dir,
                    )


        else:
            t1=time.time()
            row_wandb['transfer_time'] = t1-t0
            transfer_time = transfer_time + t1-t0
            wandb.log({**row_wandb, **sum_accuracies}, step=epoch)
            t2=time.time()


        if epoch in [args.num_epochs-1]:
            for _, worker in workers.items():
                state = {
                    'net': worker.model.state_dict(),
                    'acc': worker.valid_acc,
                    'epoch': epoch,
                }
                torch.save(state, os.path.join(worker.logdir, f'latest_{epoch}.pth'))

            img_types=['latent', 'original']
            for _img_type_ in img_types:
                for _im_id_ in range(args.num_workers):
                    imgs=(Image.open(f) for f in glob.glob(f'{fig_dir}/{_img_type_}-{_im_id_}*.png'))
                    img=next(imgs)
                    fp_out=os.path.join(fig_dir, f'_{_img_type_}-{_im_id_}.gif')
                    img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=200, loop=0)
            imgs=(Image.open(f) for f in glob.glob(f'{fig_dir}/latentmoments*.png'))
            img=next(imgs)
            fp_out=os.path.join(fig_dir, '_latentmoments.gif')
            img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=200, loop=0)
