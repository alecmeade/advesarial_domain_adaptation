import os
import numpy as np
import torch

from enum import Enum
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

DATA_DIR = 'data/'

class DatasetType(Enum):
    MNIST = 0
    USPS = 1
    SVHN = 2


def sample_dataset(data, n_samples, seed=2):
    np.random.seed(seed)
    rand_idx = np.random.permutation(len(data))
    return rand_idx[:n_samples]



def get_mnist_dataloader(train, dataloader_params, n_samples):
    transform = transforms.Compose([transforms.Resize((28, 28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))
                                   ])
    dataset = datasets.MNIST(root = DATA_DIR,
                            train = train, 
                            download = True,
                            transform = transform)

    if n_samples is None:
        return DataLoader(dataset, **dataloader_params)    

    select_idx = sample_dataset(dataset, n_samples)
    return DataLoader(dataset, sampler = SubsetRandomSampler(select_idx), **dataloader_params)


def get_usps_dataloader(train, dataloader_params, n_samples):
    transform = transforms.Compose([transforms.Resize((28, 28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))
                                    ])
    dataset = datasets.USPS(root = os.path.join(DATA_DIR, "USPS"),
                            train = train,
                            download = True,
                            transform = transform)

    if n_samples is None:
        return DataLoader(dataset, **dataloader_params)    

    select_idx = sample_dataset(dataset, n_samples)
    return DataLoader(dataset, sampler = SubsetRandomSampler(select_idx), **dataloader_params)


def get_svhn_dataloader(train, dataloader_params, n_samples):
    transform = transforms.Compose([transforms.Grayscale(),
                                    transforms.Resize((28, 28)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))
                                    ])
    dataset = datasets.SVHN(root = os.path.join(DATA_DIR, "SVHN"),
                            split = "train" if train else "test",
                            download = True,
                            transform = transform)

    if n_samples is None:
        return DataLoader(dataset, **dataloader_params)    

    select_idx = sample_dataset(dataset, n_samples)
    return DataLoader(dataset, sampler = SubsetRandomSampler(select_idx), **dataloader_params)


def get_dataloader(dataset_type, train, batch_size, n_samples=None):
    loader = None

    dataloader_params = {
        'batch_size': batch_size,
        'pin_memory': True,
    }

    if train:
        dataloader_params['drop_last'] = True

    else:
        dataloader_params['drop_last'] = False

    if dataset_type == DatasetType.MNIST:
        loader = get_mnist_dataloader(train, dataloader_params, n_samples)

    elif dataset_type == DatasetType.USPS:
        loader = get_usps_dataloader(train, dataloader_params, n_samples)

    elif dataset_type == DatasetType.SVHN:
        loader = get_svhn_dataloader(train, dataloader_params, n_samples)

    return loader
