from typing_extensions import Annotated
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from torch.utils.data import DataLoader, Sampler, Subset, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import shutil
import argparse
import random
from models import *
from worker import WorkerLtd, compute_norm
from data_utils import get_transforms, get_datasets, load_saved_embeddings, IndexSampler
from compute_divergence import train_discriminator, create_dataset_from_saved_embeddings, create_dataset_from_saved_embeddings_fedavg, SingleLayerNet

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
        transfer = f'{args.transfer}-{args.lamda}-{args.lamda_l2}-{args.mean_computation_dataset}-{args.mean_computation_batch_size}-{args.shuffle}-{share_last_layer}-{args.num_moments}'
    elif 'fedavg' in args.transfer:
        transfer = f'{args.transfer}-{args.leftout_worker_id}'
    else:
        transfer = f'{args.transfer}'

    lr = f'{args.lr}-{args.weight_decay}-{args.lr_scheduler}-{args.lr_gamma}-{args.lr_alpha}-{args.lr_decay}-{args.lrl2_gamma}-{args.lrl2_alpha}-{args.lrl2_decay}'

    name = f'{time_stamp}_{args.net}-{args.embedding_dim}_{num_workers}_{args.seed}_{lr}-{args.batch_size}_{transfer}'

    if 'synth' in args.data_path:
        name = f'{name}_synth'
    elif args.dirichlet:
        name=f'{name}_dirichlet-{args.alpha}'
    elif args.iid:
        name=f'{name}_dirichlet-'
    
    name=f'{name}_{args.use_sigmoid}_{args.num_epochs}'

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


def load_CIFAR_data():
    print("KT: in load_CIFAR_data")
    from data_utils import get_datasets, get_transforms

    transform_train, transform_test = get_transforms(grayscale=False)
    train_dataset, validation_dataset, test_dataset = get_datasets(transform_train, transform_test, args.data_path)
    transform_train_gray, transform_test_gray = get_transforms(grayscale=True)
    train_dataset_gray, validation_dataset_gray, test_dataset_gray = get_datasets(transform_train_gray, transform_test_gray, args.data_path)

    def get_partitions(file_):
        file = os.path.join(file_)
        try:
            with open(file, 'rb') as f:
                partitions = pickle.load(f)
        except:
            raise

        return partitions

    #data_file_name = f'CIFAR10-dirichlet-{args.num_workers}0-{args.alpha}-{args.data_seed}.p.p'
    #loaded_partitions = get_partitions(os.path.join('data/dirichlet-CIFAR10', data_file_name))

    if (args.iid):
        #import pdb; pdb.set_trace()
        data_file_name = f'CIFAR10-iid-partition-{args.num_workers}-{args.data_seed}.p'
        loaded_partitions = get_partitions(os.path.join('data/dirichlet-CIFAR10', data_file_name))
    elif (args.dirichlet):
        data_file_name = f'CIFAR10-dirichlet-{args.num_workers}0-{args.alpha}-{args.data_seed}.p.p'
        loaded_partitions = get_partitions(os.path.join('data/dirichlet-CIFAR10', data_file_name))
    else:
        raise NotImplementedError

    with open(os.path.join(args.logdir, 'partitions.p'), 'wb') as f:
        pickle.dump(loaded_partitions, f)

    train_subsets = dict()
    validation_subsets = dict()
    test_subsets = dict()


    # part -> worker ind
    for part in range(args.num_workers):
        train_subsets[part] = IndexSampler(indices=loaded_partitions['train'][part], shuffle=True)
        validation_subsets[part] = IndexSampler(indices=loaded_partitions['validation'][part], shuffle=True)
        test_subsets[part] = IndexSampler(indices=loaded_partitions['test'][part], shuffle=True)
        # train_subsets[part] = Subset(train_dataset, loaded_partitions['train'][part])
        # validation_subsets[part] = Subset(validation_dataset, loaded_partitions['validation'][part])
        # test_subsets[part] = Subset(test_dataset, loaded_partitions['test'][part])

    train_subsets_gray = dict()
    validation_subsets_gray = dict()
    test_subsets_gray = dict()
    for part in range(args.num_workers):
        train_subsets_gray[part] = IndexSampler(indices=loaded_partitions['train'][part], shuffle=True)
        validation_subsets_gray[part] = IndexSampler(indices=loaded_partitions['validation'][part], shuffle=True)
        test_subsets_gray[part] = IndexSampler(indices=loaded_partitions['test'][part], shuffle=True)
        # train_subsets_gray[part] = Subset(train_dataset_gray, loaded_partitions['train'][part])
        # validation_subsets_gray[part] = Subset(validation_dataset_gray, loaded_partitions['validation'][part])
        # test_subsets_gray[part] = Subset(test_dataset_gray, loaded_partitions['test'][part])


    weights=[]
    for worker_ind in range(args.num_workers):
        s=pd.Series(np.array(train_dataset.targets)[loaded_partitions['train'][worker_ind]]).value_counts()
        weights.append([s[x] if x in s.keys() else 0 for x in range(10)])
    weights = np.array(weights)
    # print(weights)
    label_counts=weights.sum(axis=0)
    weights=np.array([[weights[w,x]/label_counts[x] for x in range(10)] for w in range(args.num_workers)])
    # print(weights)

    y_train = np.array(train_dataset.targets)
    dirichlet_allocation = {}
    for x in range(10):
        dirichlet_allocation[x] = ','.join([str(sum(y_train[loaded_partitions['train'][wid]]==x)) for wid in range(args.num_workers)])

    return_val = {
        'train_dataset': train_dataset,
        'validation_dataset': validation_dataset,
        'test_dataset': test_dataset,
        'train_subsets': train_subsets,
        'validation_subsets': validation_subsets,
        'test_subsets': test_subsets,
        'train_dataset_gray': train_dataset_gray,
        'validation_dataset_gray': validation_dataset_gray,
        'test_dataset_gray': test_dataset_gray,
        'train_subsets_gray': train_subsets_gray,
        'validation_subsets_gray': validation_subsets_gray,
        'test_subsets_gray': test_subsets_gray,
        'weights': weights,
        'dirichlet_allocation': dirichlet_allocation
    }
    return return_val, 10

def load_PACS_data(train_dataset_path,
                   val_dataset_path,
                   test_dataset_path,
                   partition_path):

    from data_utils import get_datasets_pacs, get_transforms

    transform_train, transform_test = get_transforms(grayscale=False)
    train_dataset, validation_dataset, test_dataset = get_datasets_pacs(transform_train, transform_test, train_dataset_path, val_dataset_path, test_dataset_path)
    # transform_train_gray, transform_test_gray = get_transforms(grayscale=True)
    # train_dataset_gray, validation_dataset_gray, test_dataset_gray = get_datasets_pacs(transform_train_gray, transform_test_gray, train_dataset_path, val_dataset_path, test_dataset_path)

    transform_train_gray, transform_test_gray = None, None
    train_dataset_gray, validation_dataset_gray, test_dataset_gray = None, None, None

    def get_partitions(file_):
        file = os.path.join(file_)
        try:
            with open(file, 'rb') as f:
                partitions = pickle.load(f)
        except:
            raise

        return partitions
    
    loaded_partitions = get_partitions(partition_path)

    with open(os.path.join(args.logdir, 'partitions.p'), 'wb') as f:
        pickle.dump(loaded_partitions, f)

    train_subsets = dict()
    validation_subsets = dict()
    test_subsets = dict()
    # part -> worker ind
    for part in range(args.num_workers):
        train_subsets[part] = IndexSampler(indices=loaded_partitions['train'][part], shuffle=True)
        validation_subsets[part] = IndexSampler(indices=loaded_partitions['validation'][part], shuffle=True)
        test_subsets[part] = IndexSampler(indices=loaded_partitions['test'][part], shuffle=True)

    train_subsets_gray = dict()
    validation_subsets_gray = dict()
    test_subsets_gray = dict()
    '''
    for part in range(args.num_workers):
        train_subsets_gray[part] = IndexSampler(indices=loaded_partitions['train'][part], shuffle=True)
        validation_subsets_gray[part] = IndexSampler(indices=loaded_partitions['validation'][part], shuffle=True)
        test_subsets_gray[part] = IndexSampler(indices=loaded_partitions['test'][part], shuffle=True)
    '''

    weights=[]
    for worker_ind in range(args.num_workers):
        s=pd.Series(np.array(train_dataset.targets)[loaded_partitions['train'][worker_ind]]).value_counts()
        weights.append([s[x] if x in s.keys() else 0 for x in range(10)])
    weights = np.array(weights)
    label_counts=weights.sum(axis=0)
    weights=np.array([[weights[w,x]/label_counts[x] for x in range(10)] for w in range(args.num_workers)])

    y_train = np.array(train_dataset.targets)
    dirichlet_allocation = {}
    for x in range(10):
        dirichlet_allocation[x] = ','.join([str(sum(y_train[loaded_partitions['train'][wid]]==x)) for wid in range(args.num_workers)])

    return_val = {
        'train_dataset': train_dataset,
        'validation_dataset': validation_dataset,
        'test_dataset': test_dataset,
        'train_subsets': train_subsets,
        'validation_subsets': validation_subsets,
        'test_subsets': test_subsets,
        'train_dataset_gray': train_dataset_gray,
        'validation_dataset_gray': validation_dataset_gray,
        'test_dataset_gray': test_dataset_gray,
        'train_subsets_gray': train_subsets_gray,
        'validation_subsets_gray': validation_subsets_gray,
        'test_subsets_gray': test_subsets_gray,
        'weights': weights,
        'dirichlet_allocation': dirichlet_allocation
    }
    return return_val, 10

def initialize_dataset(pacs_train_path="", pacs_val_path="", pacs_test_path="", pacs_partition_path=""):
    '''
    if 'CIFAR' in args.data_path:
        print("KT loading CIFAR data")
        return_val, num_classes = load_CIFAR_data()
    el
    '''
    if 'pacs' in args.data_path.lower():
        print("KT loading PACS data")
        return_val, num_classes = load_PACS_data(pacs_train_path, pacs_val_path, pacs_test_path, pacs_partition_path)
    else:
        raise NotImplementedError

    return return_val, num_classes


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

    parser.add_argument('--gitlog', default=' ')
    parser.add_argument('--net', default='lenet', choices=['synth', 'twolayer-lenet-hetero', 'twolayer','lenet'])
    parser.add_argument('--use_sigmoid', default='relu', choices=['relu','sigmoid'])

    #DATA RELATED
    parser.add_argument('--data_type', default='PACS')
    parser.add_argument('--data_path', default='data/domain-partitioned-pacs')
    parser.add_argument('--logdir', default='./logs')
    parser.add_argument('--group_name', default='group')
    parser.add_argument('--project', default='project-ktboi')

    # PACS DATA
    # worker
    parser.add_argument('--pacs_worker_train_file', default='data/domain-partitioned-pacs/pacs-iid-serverdomain-art_painting-worker-train-split.txt')
    parser.add_argument('--pacs_worker_val_file', default='data/domain-partitioned-pacs/pacs-iid-serverdomain-art_painting-worker-val-split.txt')
    parser.add_argument('--pacs_worker_test_file', default='data/domain-partitioned-pacs/pacs-iid-serverdomain-art_painting-worker-test-split.txt')
    parser.add_argument('--pacs_worker_partition_file', default='data/domain-partitioned-pacs/pacs-iid-serverdomain-art_painting-worker-20-0.p.p')
    # server
    parser.add_argument('--pacs_server_train_file', default='data/domain-partitioned-pacs/pacs-iid-serverdomain-art_painting-server-train-split.txt')
    parser.add_argument('--pacs_server_val_file', default='data/domain-partitioned-pacs/pacs-iid-serverdomain-art_painting-server-val-split.txt')
    parser.add_argument('--pacs_server_test_file', default='data/domain-partitioned-pacs/pacs-iid-serverdomain-art_painting-server-test-split.txt')

    parser.add_argument('--embedding_dim', default=84, type=int)
    parser.add_argument('--data_seed', default=0, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--dirichlet', default=True, action='store_true')
    parser.add_argument('--iid', default=False, action='store_true')
    parser.add_argument('--alpha', default=0.1, type=float)
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

    parser.add_argument('--transfer', default='none', choices=['none', 'fedcda','fedcda-marginal', 'fedavg', 'feddf'])
    parser.add_argument('--leftout_worker_id', default=1, type=int)
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

    # randomly choose a worker to exclude in case of 'fedavg'
    # args.leftout_worker_id = np.random.choice(args.num_workers)
    fed_workers = list(range(args.num_workers))
    # fed_workers.remove(args.leftout_worker_id)
    # non_fed_worker = [args.leftout_worker_id]

    # LOGGING
    exp_name_ = experiment_name(args)
    args.logdir = os.path.join(args.logdir, exp_name_)
    os.makedirs(args.logdir)
    writer = SummaryWriter(args.logdir)
    embeddings_dir = os.path.join(args.logdir, 'embeddings')
    os.makedirs(embeddings_dir)
    wandb.init(
        project=args.project, 
        entity='ktb-throw1', 
        settings=wandb.Settings(start_method="thread"),
        name=exp_name_,
        config=args,
        group=args.group_name
    )

    time.sleep(5)

    # DATASETS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if (args.data_type == "PACS"):
        # print("KT: ", args.pacs_worker_train_file, args.pacs_worker_val_file, args.pacs_worker_test_file, args.pacs_worker_partition_file)
        loaded_dataset, num_classes = initialize_dataset(pacs_train_path=args.pacs_worker_train_file, 
                                                         pacs_val_path=args.pacs_worker_val_file, 
                                                         pacs_test_path=args.pacs_worker_test_file, 
                                                         pacs_partition_path=args.pacs_worker_partition_file)

    if 'dirichlet_allocation' in loaded_dataset:
        args.__dict__['allocation'] = loaded_dataset['dirichlet_allocation']

    if 'twolayer-lenet-hetero' in args.net:
        arch_allocation = np.random.choice(4,args.num_workers)
    else:
        arch_allocation = np.random.choice(2,args.num_workers)
    
    args.__dict__['arch_allocation'] = ','.join(list(map(str,arch_allocation)))
    with open(os.path.join(args.logdir,'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    workers = {}
    for worker_ind in range(args.num_workers):
        if (args.net == 'twolayer-lenet-hetero' and arch_allocation[worker_ind]>=2) or (args.net=='twolayer'):
            if args.dirichlet:
                temp_train_set = loaded_dataset['train_subsets_gray'][worker_ind]
                temp_valid_set = loaded_dataset['validation_subsets_gray'][worker_ind]
                temp_test_set = loaded_dataset['test_subsets_gray'][worker_ind]
                # temp_test_subset_full=loaded_dataset['test_dataset_gray']
            else:
                raise NotImplementedError
        else:
            if args.dirichlet or args.iid:
                temp_train_set = loaded_dataset['train_subsets'][worker_ind]
                temp_valid_set = loaded_dataset['validation_subsets'][worker_ind]
                temp_test_set = loaded_dataset['test_subsets'][worker_ind]
                # temp_test_subset_full=loaded_dataset['test_dataset']
            else:
                # import pdb; pdb.set_trace()
                raise NotImplementedError
                # temp_test_set = loaded_dataset['test_subset']

        # _user_train_sampler = IndexSampler(indices=self.train_partitions[i], shuffle=True)
        # _user_train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, sampler=_user_train_sampler, drop_last=False)            
        workers[worker_ind] = WorkerLtd(
            worker_ind, 
            args.net, 
            train_subset=loaded_dataset['train_dataset'], 
            train_sampler=temp_train_set,
            validation_subset=loaded_dataset['validation_dataset'], 
            validation_sampler=temp_valid_set,
            logdir=args.logdir, 
            use_sigmoid=args.use_sigmoid,
            test_subset=loaded_dataset['test_dataset'],
            test_sampler=temp_test_set,
            transfer=args.transfer,
            num_moments=args.num_moments,
            weights=loaded_dataset['weights'],
            num_workers=args.num_workers,
            lamda=args.lamda,
            lamda_l2=args.lamda_l2,
            lr=args.lr,
            weight_decay=args.weight_decay,
            lr_scheduler=args.lr_scheduler,
            lr_gamma=args.lr_gamma,
            per_layer_lr=args.per_layer_lr,
            train_batch_size=args.batch_size,
            mean_computation_batch_size=args.mean_computation_batch_size,
            normalize=args.normalize,
            hetero_arch=args.hetero_arch, 
            embedding_dim=args.embedding_dim,
            arch_allocation=arch_allocation[worker_ind],
            mean_computation_dataset=args.mean_computation_dataset,
            device=device,
            writer=writer
        )

    consensus_model = LeNetBounded(); consensus_model = consensus_model.to(device);
    temp_consensus_model = LeNetBounded(); temp_consensus_model = temp_consensus_model.to(device);
    server_model = LeNetBounded(); server_model = server_model.to(device);

    with open(os.path.join(args.logdir, f'combined_results.csv'), 'a+') as f_:
        csv_writer = csv.writer(f_,delimiter=',')
        csv_writer.writerow(['Best','Latest','BestFull','LatestFull','Validation'])

    running_best_acc = 0
    running_best_acc_full = 0
    running_best_accs = {}
    running_best_accs_full = {}
    latest_accs = {}
    latest_accs_full = {}
    for _, worker in workers.items():

        running_best_accs_full[worker.worker_ind] = running_best_acc_full
        running_best_accs[worker.worker_ind] = running_best_acc
        latest_accs_full[worker.worker_ind] = running_best_acc_full
        latest_accs[worker.worker_ind] = running_best_acc

    if 'twolayer-lenet-hetero' in args.net:
        temp_embed_dim_ = 84
    elif 'twolayer' in args.net:
        temp_embed_dim_ = 100
    elif 'lenet' in args.net:
        temp_embed_dim_ = args.embedding_dim
    
    consensus_embeddings = torch.zeros(num_classes,temp_embed_dim_,args.num_moments)
    marginal_consensus_embeddings = torch.zeros(temp_embed_dim_,args.num_moments)
    weights = torch.FloatTensor(loaded_dataset['weights']).to(device)
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
    infer_time_ts = 0
    infer_time_val = 0

    transfer_time = 0

    for epoch in tqdm(range(args.num_epochs)):
        ep_train_time = 0
        ep_log_time = 0
        ep_infer_time_ts = 0
        ep_infer_time_val = 0
        
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
            t2v=time.time()
            ep_infer_time_val = ep_infer_time_val + t2v-t1
            worker.inference(epoch, valid=False)
            t2=time.time()
            ep_infer_time_ts = ep_infer_time_ts + t2-t2v

            latest_accs[worker.worker_ind] = worker.test_acc
            latest_accs_full[worker.worker_ind] = worker.test_acc_full
            if worker.valid_acc == worker.best_acc:
                running_best_acc_full = worker.test_acc_full
                running_best_accs_full[worker.worker_ind] = running_best_acc_full
                running_best_acc = worker.test_acc
                running_best_accs[worker.worker_ind] = running_best_acc

            row_wandb[f'best_test_acc_full_{worker.worker_ind}'] = running_best_accs_full[worker.worker_ind]
            row_wandb[f'best_test_acc_{worker.worker_ind}'] = running_best_accs[worker.worker_ind]
            row_wandb[f'train_acc_{worker.worker_ind}'] = worker.train_acc
            row_wandb[f'test_acc_{worker.worker_ind}'] = worker.test_acc
            row_wandb[f'test_acc_full_{worker.worker_ind}'] = worker.test_acc_full
            row_wandb[f'valid_acc_{worker.worker_ind}'] = worker.valid_acc
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

            row_wandb[f'Reg_{worker.worker_ind}'] = regularization_losses[f'total_loss']/batch_idx
            row_wandb[f'Reg-L2_{worker.worker_ind}'] = regularization_losses[f'l2_reg']/batch_idx
            for moment_ in range(1,args.num_moments+1):
                row_wandb[f'Reg_M{moment_}_{worker.worker_ind}'] = regularization_losses[f'moment_{moment_}_loss']/batch_idx

            t3=time.time()
            ep_log_time = ep_log_time + t3-t2
        
        train_time += ep_train_time
        log_time += ep_log_time
        infer_time_ts += ep_infer_time_ts
        infer_time_val += ep_infer_time_val

        row_wandb['ep_train_time'] = ep_train_time
        row_wandb['ep_log_time'] = ep_log_time
        row_wandb['ep_infer_time_ts'] = ep_infer_time_ts
        row_wandb['ep_infer_time_val'] = ep_infer_time_val

        mean_accuracy = 0
        accuracies = [running_best_accs[worker_ind_] for worker_ind_ in running_best_accs.keys()]
        latest_accuracies = [latest_accs[worker_ind_] for worker_ind_ in latest_accs.keys()]
        accuracies_full = [running_best_accs_full[worker_ind_] for worker_ind_ in running_best_accs_full.keys()]
        latest_accuracies_full = [latest_accs_full[worker_ind_] for worker_ind_ in latest_accs_full.keys()]
        valid_accuracies = [workers[_ind_].valid_acc for _ind_ in range(args.num_workers)]

        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        mean_latest_accuracy = np.mean(latest_accuracies)
        std_latest_accuracy = np.std(latest_accuracies)
        mean_accuracy_full = np.mean(accuracies_full)
        std_accuracy_full = np.std(accuracies_full)
        mean_latest_accuracy_full = np.mean(latest_accuracies_full)
        std_latest_accuracy_full = np.std(latest_accuracies_full)

        mean_valid_accuracy = np.mean(valid_accuracies)
        std_valid_accuracy = np.std(valid_accuracies)

        sum_accuracies={
            'mean_best_accuracy': mean_accuracy,
            'std_best_accuracy': std_accuracy,
            'mean_latest_accuracy': mean_latest_accuracy,
            'std_latest_accuracy': std_latest_accuracy,
            'mean_best_accuracy_full': mean_accuracy_full,
            'std_best_accuracy_full': std_accuracy_full,
            'mean_latest_accuracy_full': mean_latest_accuracy_full,
            'std_latest_accuracy_full': std_latest_accuracy_full,
            'mean_valid_accuracy': mean_valid_accuracy,
            'std_valid_accuracy': std_valid_accuracy
        }
        combined_res_row = [str(mean_accuracy), str(mean_latest_accuracy),str(mean_accuracy_full), str(mean_latest_accuracy_full), str(mean_valid_accuracy)]
        with open(os.path.join(args.logdir, f'combined_results.csv'), 'a+') as f_:
            csv_writer = csv.writer(f_,delimiter=',')
            csv_writer.writerow(combined_res_row)

        if args.hetero_arch:
            sum_accuracies['mean_best_accuracy_arch-0']= np.mean([running_best_accs[worker_ind_] for worker_ind_ in range(args.num_workers) if arch_allocation[worker_ind_]==0])
            sum_accuracies['mean_best_accuracy_arch-1']= np.mean([running_best_accs[worker_ind_] for worker_ind_ in range(args.num_workers) if arch_allocation[worker_ind_]==1])

        ########## Communication ############

        # Transfer knowledge here
        t0=time.time()

        if 'marginal' in args.transfer:
            with torch.no_grad():
                marginal_moment_embeddings = torch.zeros(args.num_workers, worker.embedding_dim, args.num_moments)
                for _, worker in workers.items():
                    shuffle_ = args.shuffle == 'True'
                    save_embeddings_ = ((epoch % args.discr_comp_interval==0) or (epoch==args.num_epochs-1)) and args.compute_divergence
                    temp_embed = worker.compute_marginal_moment_embeddings(epoch=epoch, logdir=args.logdir, save_embeddings=save_embeddings_, shuffle=shuffle_)

                    # Update after checking if the embeddings are not zeros (initialized values)
                    if not (temp_embed[:,:] == 0).all():
                        marginal_moment_embeddings[worker.worker_ind,:,:] = temp_embed[:,:]

                if torch.isnan(marginal_moment_embeddings).any():
                    print('marginal_moment_embeddings')
                    raise

            assert not (marginal_moment_embeddings == 0).all()
            with torch.no_grad():
                marginal_moment_embeddings=marginal_moment_embeddings.to(device)
                marginal_consensus_embeddings=marginal_consensus_embeddings.to(device)
                alpha_t = anneal_cons_alpha(args, epoch, l2=False)
                cons_update_norm = 0
                _cons_emb_=torch.einsum('b, bij -> ij', weights.sum(axis=1)/weights.sum(), marginal_moment_embeddings[:,:,:])
                cons_update_norm += torch.norm(marginal_consensus_embeddings[:,:] - _cons_emb_)**2
                marginal_consensus_embeddings[:,:] = (1-alpha_t) * marginal_consensus_embeddings[:,:] + alpha_t * _cons_emb_
                cons_update_norm = torch.sqrt(cons_update_norm)
                row_wandb['consensus_update_norm'] = cons_update_norm

                for _, worker in workers.items():
                    worker.store_cons_marginal_moment_embeddings(marginal_consensus_embeddings)
                
                np.save(os.path.join(embeddings_dir, f'epoch-{epoch}.npy'), np.array(marginal_consensus_embeddings.cpu()))

        else:
            with torch.no_grad():
                moment_embeddings = torch.zeros(args.num_workers, num_classes, worker.embedding_dim, args.num_moments)
                for _, worker in workers.items():
                    shuffle_ = args.shuffle == 'True'
                    save_embeddings_ = ((epoch % args.discr_comp_interval==0) or (epoch==args.num_epochs-1)) and args.compute_divergence
                    temp_embed = worker.compute_moment_embeddings(epoch=epoch, logdir=args.logdir, save_embeddings=save_embeddings_, shuffle=shuffle_)

                    for temp_lab in range(num_classes):
                        # Update after checking if the embeddings are not zeros (initialized values)
                        if not (temp_embed[temp_lab,:,:] == 0).all():
                            moment_embeddings[worker.worker_ind,temp_lab,:,:] = temp_embed[temp_lab,:,:]

                if torch.isnan(moment_embeddings).any():
                    print('moment embeddings')
                    raise

            assert not (moment_embeddings == 0).all()
            with torch.no_grad():
                moment_embeddings=moment_embeddings.to(device)
                consensus_embeddings=consensus_embeddings.to(device)
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

        if (args.transfer != 'none') and args.share_last_layer: # how to share parameter weights
            # import pdb; pdb.set_trace()
            with torch.no_grad():
                for p1, p2 in zip(temp_consensus_classifier.parameters(), workers[0].model.classifier.parameters()):
                    if args.weighted_sharing:
                        p1.data = p2.detach().clone().data * (weights.sum(axis=1)[0]/weights.sum())
                    else:
                        p1.data = p2.detach().clone().data/args.num_workers

                for wid in range(1, args.num_workers):
                    worker = workers[wid]
                    for p1, p2 in zip(temp_consensus_classifier.parameters(), worker.model.classifier.parameters()):
                        if args.weighted_sharing:
                            p1.data = p1.data + (weights.sum(axis=1)[wid]/weights.sum()) * p2.detach().clone().data
                        else:
                            p1.data = p1.data + p2.detach().clone().data/args.num_workers

                cons_classifier_norm = compute_norm([p1.detach().clone() - p2.detach().clone() for (p1,p2) in zip(consensus_classifier.parameters(),temp_consensus_classifier.parameters())])
                row_wandb['consensus_classifier_norm'] = cons_classifier_norm

                alpha_t = anneal_cons_alpha(args, epoch, l2=True, discr=discr_accuracy_total)
                if 'fedcda' in args.transfer:
                    for p1, p2 in zip(consensus_classifier.parameters(), temp_consensus_classifier.parameters()):
                        p1.data = p1.data* (1-alpha_t) + alpha_t* p2.detach().clone().data

                    for wid, worker in workers.items():
                        worker.store_global_classifier(consensus_classifier)
        
        elif (args.transfer == 'fedavg'):
            # import pdb; pdb.set_trace()
            with torch.no_grad():
                workers_processed = set()
                if args.leftout_worker_id == 0:
                    for p1, p2 in zip(temp_consensus_model.parameters(), workers[1].model.parameters()):
                        p1.data = p2.detach().clone().data/min(args.num_workers - 1, 1)
                    workers_processed.add(0)
                else:
                    for p1, p2 in zip(temp_consensus_model.parameters(), workers[0].model.parameters()):
                        p1.data = p2.detach().clone().data/min(args.num_workers - 1, 1)
                    workers_processed.add(1)

                for wid in range(args.num_workers):
                    if wid in workers_processed:
                        continue
                    worker = workers[wid]
                    for p1, p2 in zip(temp_consensus_model.parameters(), worker.model.parameters()):
                        p1.data = p1.data + p2.detach().clone().data/min(args.num_workers - 1, 1)

                cons_model_norm = compute_norm([p1.detach().clone() - p2.detach().clone() for (p1,p2) in zip(consensus_model.parameters(),temp_consensus_model.parameters())])
                row_wandb['consensus_model_norm'] = cons_model_norm

                for wid in range(args.num_workers):
                    if args.leftout_worker_id == wid:
                        continue
                    worker = workers[wid]
                    for p1, p2 in zip(consensus_model.parameters(), worker.model.parameters()):
                        p2.data = p1.detach().clone().data

        elif (args.transfer == 'feddf'):
        # tempConsensusModel captures average of the worker model parameters
        # serverModel starts from tempConsensusModel and changes it by back proping on noisy data.
        # this changed serverModel is sent over to the worker.
            FedDFEpochs = 10;
            with torch.no_grad():
                for p1, p2, p3 in zip(temp_consensus_model.parameters(), workers[0].model.parameters(), server_model.parameters()):
                    p1.data = p2.detach().clone().data/args.num_workers
                    p3.data = p2.detach().clone().data/args.num_workers

                for wid in range(1, args.num_workers):
                    worker = workers[wid]
                    for p1, p2, p3 in zip(temp_consensus_model.parameters(), worker.model.parameters(),  server_model.parameters()):
                        p1.data = p1.data + p2.detach().clone().data/args.num_workers
                        p3.data = p1.data + p2.detach().clone().data/args.num_workers

            # get noisy data
            from data_utils import get_datasets, get_datasets_pacs, get_transforms

            '''
            if (args.data_type == "PACS"):
                noisey_transform_train, noisey_transform_test = get_transforms(grayscale=False)
                noisey_train_dataset, noisey_validation_dataset, noisey_test_dataset = get_datasets_pacs(noisey_transform_train, noisey_transform_test, 
                                                                                                         args.pacs_server_train_file, args.pacs_server_val_file, args.pacs_server_test_file) 
            else:
                noisey_transform_train, noisey_transform_test = get_transforms(grayscale=False, noise_mean=1, noise_std=1)
                noisey_train_dataset, noisey_validation_dataset, noisey_test_dataset = get_datasets(noisey_transform_train, noisey_transform_test, args.data_path)
            '''
            noisey_transform_train, noisey_transform_test = get_transforms(grayscale=False)
            noisey_train_dataset, noisey_validation_dataset, noisey_test_dataset = get_datasets_pacs(noisey_transform_train, noisey_transform_test, 
                                                                                                         args.pacs_server_train_file, args.pacs_server_val_file, args.pacs_server_test_file) 
            
            # Create data loaders for our datasets; shuffle for training, not for validation
            noisey_training_loader = torch.utils.data.DataLoader(noisey_train_dataset, batch_size=16, shuffle=True)
            noisey_validation_loader = torch.utils.data.DataLoader(noisey_validation_dataset, batch_size=16, shuffle=False)

            server_loss_fn = torch.nn.MSELoss()
            server_optimizer = torch.optim.SGD(server_model.parameters(), lr = 0.01, momentum = 0.9)
            # TODO:
            # get noisy data.
            # train server_model
            # send server_model to workers
            for server_epoch in tqdm(range(FedDFEpochs)):
                temp_consensus_model.eval()
                server_model.train()
                feddf_running_loss = 0.0
                feddf_last_loss = 0.0
                
                for server_idx, noisey_data in enumerate(noisey_training_loader):

                    server_optimizer.zero_grad()

                    noisey_inputs, _ = noisey_data
                    noisey_inputs = noisey_inputs.to('cuda')

                    consensus_model_logits = temp_consensus_model(noisey_inputs)
                    server_model_logits = server_model(noisey_inputs)

                    # import pdb; pdb.set_trace()

                    loss_val = server_loss_fn(server_model_logits[1], consensus_model_logits[1])
                    loss_val.backward()

                    server_optimizer.step()

                    feddf_running_loss += loss_val.item()

                    if server_idx % 10 == 9:
                        feddf_last_loss = feddf_running_loss / 10
                        row_wandb['feddf_batch{}_epoch{}_loss'.format(server_idx, server_epoch, feddf_last_loss)] = feddf_last_loss
                        feddf_running_loss = 0.0
            
            # update each worker's parameters
            with torch.no_grad():
                for p1, p2 in zip(server_model.parameters(), workers[0].model.parameters()):
                    p2.data = p1.detach().clone().data

                for wid in range(1, args.num_workers):
                    worker = workers[wid]
                    for p1, p2 in zip(server_model.parameters(), worker.model.parameters()):
                        p2.data = p1.detach().clone().data
        
        # TODO
        # elif (args.transfer == 'fedDf'):
        # loop over each student's model.
        # compute the average of each model architecture
        # for each model architecture
        #   for each epoch
        #     for each batch ---> add noise specified by alpha to each image in the batch
        #       backward prop/train
        #   update model architecture to send to student.
        # NOTE: student data does not contain noise.
        
        ###################################################
        # Compute divergence
        if ((epoch % args.discr_comp_interval==0) or (epoch==args.num_epochs-1)) and args.compute_divergence:
            
            discr = {}

            '''
            loaded_dataset = create_dataset_from_saved_embeddings(
                load_saved_embeddings(os.path.join(args.logdir, 'saved_embeddings'), epoch=epoch, workers=list(range(args.num_workers))),
                labels=list(range(num_classes)),
                workers=list(range(args.num_workers))
            )
            '''

            loaded_dataset = create_dataset_from_saved_embeddings_fedavg(
                load_saved_embeddings(os.path.join(args.logdir, 'saved_embeddings'), epoch=epoch, workers=list(range(args.num_workers))), fed_workers, non_fed_worker)

            discr_batch_size = 128

            loader = DataLoader(
                loaded_dataset, batch_size=discr_batch_size, shuffle=True,
                pin_memory=True
            )
            model = SingleLayerNet(num_workers=args.num_workers, hsize=temp_embed_dim_).to(device)
            discrim_accuracy, _ = train_discriminator(model, loader, device=device, num_epochs=100, lr=0.01,writer=writer, data_epoch=epoch)
            discr[f'discr_accuracy_total'] = discrim_accuracy
            discr_accuracy_total = discrim_accuracy/100

            wandb.log({**row_wandb, **sum_accuracies, **discr}, step=epoch)

            # Delete the saved_embeddings epoch files. 
            shutil.rmtree(os.path.join(args.logdir,'saved_embeddings'))

        else:

            wandb.log({**row_wandb, **sum_accuracies}, step=epoch)
        t1=time.time()
        # row_wandb['transfer_time'] = t1-t0
        transfer_time = transfer_time + t1-t0

        if epoch in [0, 10, 50, 100, 150, args.num_epochs-1]:
            for _, worker in workers.items():
                state = {
                    'net': worker.model.state_dict(),
                    'acc': worker.valid_acc,
                    'epoch': epoch,
                }
                torch.save(state, os.path.join(worker.logdir, f'latest_{epoch}.pth'))

