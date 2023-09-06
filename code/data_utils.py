import os
from matplotlib.pyplot import gray
import torch
from torchvision import transforms
from torchvision import datasets


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
    

def get_transforms(grayscale=False):

    if not grayscale:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,), (0.2023,)),
        ])
        transform_test = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,), (0.2023,)),
        ])


    return transform_train, transform_test

def get_datasets(transform_train, transform_test, data_path):
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

