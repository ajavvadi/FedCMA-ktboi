
import os
import torch
import pandas as pd
from PIL import Image
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class PACSDataset(Dataset):
    """PACS dataset."""

    def __init__(self, text_file, transform=None):
        if not os.path.exists(text_file):
            print("{} does not exist")
            exit(1)

        self.data = []
        self.targets = []
        self.targets_name = []
        self.all_files = []
        with open(text_file, 'r') as f_h:
            self.all_files = f_h.readlines()
            for each_file in self.all_files:
                imgpath, labelid, labelname = each_file.split(", ")
                self.data.append(io.imread(imgpath))
                assert self.data[-1].shape == (227, 227, 3), "image shape: {}".format(self.data[-1].shape)
                self.targets.append(int(labelid))
                self.targets_name.append(labelname)
        self.transform = transform

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_arr, label_id = self.data[idx], self.targets[idx]
        
        image = Image.fromarray(img_arr)

        if self.transform:
            image = self.transform(image)

        # we don't need to transfor label_id because we will transform it in the training loop

        return image, label_id









































































def load_pacs_data():
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

