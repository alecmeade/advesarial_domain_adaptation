import torch.nn as nn

class LeNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(3, 6, kernel_size = 5),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2, stride = 2),
			nn.Conv2d(6, 16, kernel_size = 5),
			nn.MaxPool2d(kernel_size = 2, stride = 2),
			nn.ReLU(),
			nn.Conv2d(16, 120, kernel_size = 5),
		)

		self.classifier = nn.Sequential(
			nn.Linear(120, 84),
			nn.ReLU(),
			nn.Linear(84, 10)
		)

	def forward(self, x):
		x = self.encoder(x)
		x = x.view(x.shape[0], -1)
		return self.classifier(x)