import torch.nn

class AddaDiscriminator(nn.Module):
	def __init__(self, out_features = 1):
		super().__init__()
		self.model = Sequential(
	        nn.Linear(500, 500),
	        nn.ReLU(),
	        nn.Linear(500, 500),
	        nn.ReLU(),
	        nn.Linear(500, out_features)
		)

	def forward(self, x):
		return self.model()