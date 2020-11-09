import argparse
import cuda_utils
import torch
import torch.nn as nn

from dataloaders import DatasetType, get_dataloader
from lenet import LeNet


def source_train(args):
	dataloader_params = {
		'batch_size': args.batch_size,
		'pin_memory': True,
		'drop_last': True,
	}

	device = cuda_utils.maybe_get_cuda_device()

	loader = get_dataloader(DatasetType[args.source_dataset], 
							True, 
					        dataloader_params, 
					        args.n_samples)

	model = LeNet()
	model.to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)

	for x_batch, y_batch in loader:
		x_batch = x_batch.to(device)
		y_batch = y_batch.to(device)
		out = model(x_batch)
		print(out.shape)
		break

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--source_dataset', type=str, default = "MNIST")
    arg_parser.add_argument('--n_samples', type=int, default = 7000)
    arg_parser.add_argument('--learning_rate', type=float, default = 0.01)
    arg_parser.add_argument('--batch_size', type=int, default = 16) 
    arg_parser.add_argument('--epochs', type=int, default = 30)
    args = arg_parser.parse_args()

    source_train(args)