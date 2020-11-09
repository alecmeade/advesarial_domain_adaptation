import os
import numpy as np
import torch

from enum import Enum
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

DATA_DIR = 'data/'

class DatasetType(Enum):
    MNIST = 0
    USPS = 1
    SVHN = 2

def sample_dataset(data, n_samples, seed):
    np.random.seed(seed)
    rand_idx = np.random.permutation(len(data))

    return rand_idx[:n_samples]

def get_mnist_dataloader(train, dataloader_params, n_samples, seed):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))
                                   ])
    dataset = datasets.MNIST(root = DATA_DIR,
                            train = train, 
                            download = True,
                            transform = transform)

    if n_samples is None:
        return DataLoader(dataset, **dataloader_params)    

    select_idx = sample_dataset(dataset, n_samples, seed)
    return DataLoader(dataset, sampler = SubsetRandomSampler(select_idx), **dataloader_params)


def get_usps_dataloader(train, dataloader_params, n_samples, seed):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, ), (0.5,))
                                    ])
    dataset = datasets.USPS(root = os.path.join(DATA_DIR, "USPS"),
                            train = train,
                            download = True,
                            transform = transform)

    if n_samples is None:
        return DataLoader(dataset, **dataloader_params)    

    select_idx = sample_dataset(dataset, n_samples, seed)
    return DataLoader(dataset, sampler = SubsetRandomSampler(select_idx), **dataloader_params)


def get_svhn_dataloader(train, dataloader_params, n_samples, seed):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
                                    ])
    dataset = datasets.SVHN(root = os.path.join(DATA_DIR, "SVHN"),
                            split = "train" if train else "test",
                            download = True,
                            transform = transform)

    if n_samples is None:
        return DataLoader(dataset, **dataloader_params)    

    select_idx = sample_dataset(dataset, n_samples, seed)
    return DataLoader(dataset, sampler = SubsetRandomSampler(select_idx), **dataloader_params)


def get_dataloader(dataset_type, train, dataloader_params, n_samples=None, seed=0):
    loader = None

    if dataset_type == DatasetType.MNIST:
        loader = get_mnist_dataloader(train, dataloader_params, n_samples, seed)

    elif dataset_type == DatasetType.USPS:
        loader = get_usps_dataloader(train, dataloader_params, n_samples, seed)

    elif dataset_type == DatasetType.SVHN:
        loader = get_svhn_dataloader(train, dataloader_params, n_samples, seed)

    return loader
