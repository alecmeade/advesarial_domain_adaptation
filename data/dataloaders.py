import os
import torch
from torchivision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_dataloader(dataset_dir, train, dataloader_params):
	mnist_mu = 0.1307
	mnist_stdev = 0.3081
	transform = transforms.Compose([transforms.ToTensor(),
				                    transforms.Normalize((mnist_mu,), (mnist_stdev,))])
	dataset = datasets.MNIST(root = os.path.join(dataset_dir, "MNSIT"),
							 train = train, 
							 download = True,
                         	 transform = transform)

    return DataLoader(dataset, **dataloader_params)

def get_usps_dataloader(dataset_dir, train, dataloader_params):
	pass


def get_svhn_dataloader(dataset_dir, train, dataloader_params):
	pass