import os
from matplotlib.pyplot import gray
import torch
from torchvision import transforms
from torchvision import datasets
from pacs_data_util import PACSDataset


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class IndexSampler(object):
    def __init__(self, indices, shuffle=True) -> None:
        # super().__init__()
        self.g=torch.Generator()
        self.g.manual_seed(0)
        self.indices = indices
        self.shuffle = shuffle
        self.num_samples = len(self.indices)

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(len(self.indices), generator=self.g).tolist()
        else:
            indices = self.indices

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def __str__(self) -> str:
        return f'IndexSampler'
    

def get_transforms(grayscale=False, noise_mean = 0, noise_std = 1):

    if not grayscale:
        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            AddGaussianNoise(noise_mean, noise_std)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,), (0.2023,)),
            AddGaussianNoise(noise_mean, noise_std)
        ])
        transform_test = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,), (0.2023,)),
        ])


    return transform_train, transform_test

def get_datasets(transform_train, transform_test, data_path):
    print("KT: in get_datasets")
    train_dataset = datasets.CIFAR10(
        data_path,
        train=True,
        download=True,
        transform=transform_train
    )
    validation_dataset = datasets.CIFAR10(
        data_path,
        train=True,
        download=True,
        transform=transform_test
    )
    test_dataset = datasets.CIFAR10(
        data_path,
        train=False,
        download=True,
        transform=transform_test
    )

    return train_dataset, validation_dataset, test_dataset

def get_datasets_pacs(transform_train, transform_test, 
                      train_dataset_path,
                      val_dataset_path,
                      test_dataset_path):
    train_dataset = PACSDataset(
        train_dataset_path,
        transform=transform_train
    )
    validation_dataset = PACSDataset(
        val_dataset_path,
        transform=transform_test
    )
    test_dataset = PACSDataset(
        test_dataset_path,
        transform=transform_test
    )

    return train_dataset, validation_dataset, test_dataset

def load_saved_embeddings(folder, epoch, workers):
    """
    Outputs a dict of all the workers' saved embeddings along with labels
    """

    loaded_data = {}
    for worker in workers:

        file_name = os.path.join(folder, f'{worker}-{epoch}.pth')
        loaded = torch.load(file_name)

        loaded_data[worker] = loaded

    return loaded_data

