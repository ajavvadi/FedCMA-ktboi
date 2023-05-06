import os
from matplotlib.pyplot import gray
import torch
from torchvision import transforms
from torchvision import datasets

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

