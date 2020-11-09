import os
import numpy as np
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def get_mnist_dataloader(dataset_dir, train, dataloader_params, dist_mean=0.5, dist_stddev=0.5):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((dist_mean,), (dist_stddev,))])
    dataset = datasets.MNIST(root = dataset_dir,
                            train = train, 
                            download = True,
                            transform = transform)

    return DataLoader(dataset, **dataloader_params)


def get_usps_dataloader(dataset_dir, train, dataloader_params, dist_mean=0.5, dist_stddev=0.5):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((dist_mean,), (dist_stddev,))])
    dataset = datasets.USPS(root = os.path.join(dataset_dir, "USPS"),
                            train = train,
                            download = True,
                            transform = transform)

    return DataLoader(dataset, **dataloader_params)


def get_svhn_dataloader(dataset_dir, train, dataloader_params, dist_mean=0.5, dist_stddev=0.5):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((dist_mean,), (dist_stddev,))])
    dataset = datasets.SVHN(root = os.path.join(dataset_dir, "SVHN"),
                            download = True,
                            transform = transform)

    # Create consistent test train splits.
    N = len(dataset)
    train_test_split = 0.7
    pivot = int(train_test_split * N)
    np.random.seed(0)
    rand_idx = np.random.permutation(N)
    select_idx = None
    
    if train:
        select_idx = rand_idx[:pivot] 

    else:
        select_idx = rand_idx[pivot:]

    return DataLoader(dataset, sampler = SubsetRandomSampler(select_idx), **dataloader_params)
