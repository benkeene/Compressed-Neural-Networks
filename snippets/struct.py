self.linear_relu_stack = nn.Sequential(
	nn.Linear(28*28, 15),
	nn.ReLU(),
	nn.Linear(15, 15),
	nn.ReLU(),
	nn.Linear(15, 10)
)