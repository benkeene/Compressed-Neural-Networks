import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from compressed_SGD import SGD
from sklearn.utils import extmath
import numpy as np


# Download training data from open datasets.
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

print(training_data)

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            #nn.Linear(28*28, 512),
            nn.Linear(28*28, 10),
            nn.ReLU(),
            nn.Linear(10, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=1e-3)


def truncSVD(model, q1, q2, q3):
    sd = model.state_dict()

    U, S, Vh = torch.linalg.svd(
        sd['linear_relu_stack.0.weight'], full_matrices=False)
    print(S)
    quit()


def rSVD(model, q):
    sd = model.state_dict()

    stack0 = torch.svd_lowrank(sd['linear_relu_stack.0.weight'], q)
    stack2 = torch.svd_lowrank(sd['linear_relu_stack.2.weight'], q)
    stack4 = torch.svd_lowrank(sd['linear_relu_stack.4.weight'], q)

    sd['linear_relu_stack.0.weight'] = (
        stack0[0] @ torch.diagflat(stack0[1]) @ torch.transpose(stack0[2], 0, 1))

    sd['linear_relu_stack.2.weight'] = (
        stack2[0] @ torch.diagflat(stack2[1]) @ torch.transpose(stack2[2], 0, 1))

    sd['linear_relu_stack.4.weight'] = (
        stack4[0] @ torch.diagflat(stack4[1]) @ torch.transpose(stack4[2], 0, 1))

    model.load_state_dict(sd)
    return


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()  # tell model we are training https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch
    for batch, (X, y) in enumerate(dataloader):  # supply model with batches of data and labels
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        if 0:
            sdim = 10
            for i, weight in enumerate(model.linear_relu_stack):
                # print("Printing weight")
                if str(weight) != 'ReLU()':
                    weight = model.linear_relu_stack[i].weight
                    weight = weight.detach().numpy()
                    weight = extmath.randomized_svd(weight, sdim)
                    U = weight[0]
                    S = np.zeros([sdim, sdim])
                    for j in range(sdim):
                        S[j, j] = weight[1][j]
                    V = weight[2]

                    temp = (U @ S @ V)

                    model.linear_relu_stack[i].weight = nn.Parameter(
                        torch.from_numpy(temp.astype('float32')))
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #rSVD(model, 10)
        truncSVD(model, 10, 20, 30)

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # Print model's state_dict
    print("\nModel's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()
              [param_tensor].size())

    print()


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()  # tell model we are evaluating https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
