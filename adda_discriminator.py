import torch.nn

class AddaDiscriminator(nn.Module):
	def __init__(self, hidden = 500, out = 1):
		super().__init__()
		self.model = Sequential(
	        nn.Linear(120, hidden),
	        nn.ReLU(),
	        nn.Linear(hidden, hidden),
	        nn.ReLU(),
	        nn.Linear(hidden, out)
		)

	def forward(self, x):
		return self.model()