import torch.nn as nn

class LeNet(nn.Module):
	def __init__(self):
		self.encoder = nn.Sequential(
			nn.Conv2d(1, 6, kernel_size = 5),
			nn.ReLU(),
			nn.MaxPool2d(2, kernel_size = 2),
			nn.Conv2d(6, 16, kernel_size = 5),
			nn.ReLU()
		)

		self.classifier = nn.Sequential(
			nn.Linear(120, 84),
			nn.ReLU(),
			nn.Linear(84, 10)
		)

	def forward(self, x):
		x = self.encoder(x)
		x = self.classifier(x)
		return x