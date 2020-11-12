import torch.nn as nn


class LeNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(1, 20, kernel_size = 5),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(20, 50, kernel_size = 5),
			nn.MaxPool2d(2),
			nn.ReLU(),
		)

		self.classifier = nn.Sequential(
			nn.Linear(800, 500),
			nn.ReLU(),
			nn.Linear(500, 10)
		)

	def forward(self, x):
		x = self.encoder(x)
		x = x.view(x.shape[0], -1)
		x = self.classifier(x)
		return x


class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()
		self.model = nn.Sequential(
	        nn.Linear(800, 500),
	        nn.ReLU(),
	        nn.Linear(500, 500),
	        nn.ReLU(),
	        nn.Linear(500, 2),
		)

	def forward(self, x):
		return self.model(x)