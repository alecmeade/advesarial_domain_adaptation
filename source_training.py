import argparse
from dataloaders import DatasetType, get_dataloader
import torch


def train(args):

	dataloader_params = {
		'batch_size': args.batche_size,
		'pin_memory': True,
		'drop_last': True
	}

	loader = get_dataloader(DatasetType[args.dataset], True, dataloader_params, args.n_samples)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset', type=str, default = "MNIST")
    arg_parser.add_argument('--n_samples', type=int, default = 7000)
    arg_parser.add_argument('--batche_size', type=int, default = 16)
    arg_parser.add_argument('--epochs', type=int, default = 30)
    args = arg_parser.parse_args()
    train(args)