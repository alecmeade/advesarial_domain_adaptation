import torch.nn as nn

class LeNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(3, 6, 5),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			nn.Conv2d(6, 16, 5),
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