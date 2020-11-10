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

	train_loader = get_dataloader(DatasetType[args.source_dataset], 
								  True, 
							      dataloader_params, 
							      (args.img_dim, args.img_dim),
							      args.n_samples)


	dataloader_params['drop_last'] = False
	eval_loader = get_dataloader(DatasetType[args.source_dataset], 
								  False, 
							      dataloader_params, 
							      (args.img_dim, args.img_dim),
							      args.n_samples)

	model = LeNet()
	model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
	criterion = nn.CrossEntropyLoss()

	for epoch in range(args.epochs):
		# Train
		model.train()
		torch.grad()
		for x_batch, y_batch in train_loader:
			train_step(model, optimizer, criterion, x_batch, y_batch)

		# Evaluate
		model.eval()
		torch.no_grad()
		for x_batch, y_batch in eval_loader:
        	source_eval(model, criterion, x_batch, y_batch)


    return model


def train_step(model, optimizer, criterion, x_batch, y_batch):
	x_batch = x_batch.to(device)
	y_batch = y_batch.to(device)

	optimizer.zero_grad()

	out = model(x_batch)
	loss = criterion(out, y_batch)
	_, pred = torch.max(out, 1)

	loss.backward()
    optimizer.step()


def eval_step(model, criterion, x_batch, y_batch):
	x_batch = x_batch.to(device)
	y_batch = y_batch.to(device)

	out = model(x_batch)
	loss = criterion(out, y_batch)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--source_dataset', type=str, default = "MNIST")
    arg_parser.add_argument('--n_samples', type=int, default = 7000)
    arg_parser.add_argument('--img_dim', type=int, default = 32)
    arg_parser.add_argument('--learning_rate', type=float, default = 0.01)
    arg_parser.add_argument('--batch_size', type=int, default = 16) 
    arg_parser.add_argument('--epochs', type=int, default = 30)
    args = arg_parser.parse_args()

    model = source_train(args)