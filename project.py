import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from compressed_SGD import SGD
from sklearn.utils import extmath
import numpy as np
import timeit
import matplotlib.pyplot as plt


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
            nn.Linear(28*28, 15),
            nn.ReLU(),
            nn.Linear(15, 15),
            nn.ReLU(),
            nn.Linear(15, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)


def truncSVD(model, q1):
    sd = model.state_dict()

    U0, S0, Vh0 = torch.linalg.svd(
        sd['linear_relu_stack.0.weight'], full_matrices=False)
    U2, S2, Vh2 = torch.linalg.svd(
        sd['linear_relu_stack.2.weight'], full_matrices=False)
    U4, S4, Vh4 = torch.linalg.svd(
        sd['linear_relu_stack.4.weight'], full_matrices=False)

    for j in range(15):
        if j > q1:
            S0[j] = 0
        if j > q1:
            S2[j] = 0
    for j in range(10):
        if j > q1:
            S4[j] = 0

    sd['linear_relu_stack.0.weight'] = torch.matmul(
        torch.matmul(U0, torch.diag(S0)), Vh0)
    sd['linear_relu_stack.2.weight'] = torch.matmul(
        torch.matmul(U2, torch.diag(S2)), Vh2)
    sd['linear_relu_stack.4.weight'] = torch.matmul(
        torch.matmul(U4, torch.diag(S4)), Vh4)

    model.load_state_dict(sd)
    return


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


def train(dataloader, model, loss_fn, optimizer, rsvd_flag, truncsvd_flag, rsvd_param=10, truncsvd_param=10):
    size = len(dataloader.dataset)
    model.train()  # tell model we are training https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch
    for batch, (X, y) in enumerate(dataloader):  # supply model with batches of data and labels
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if rsvd_flag:
            rSVD(model, rsvd_param)
        if truncsvd_flag:
            truncSVD(model, truncsvd_param)

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
    return 100*correct, test_loss


def figure_1():
    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=1e-2)
    acc = []
    loss = []

    epochs = 10
    indexed_epochs = [x+1 for x in range(epochs)]
    start = timeit.default_timer()
    for t in indexed_epochs:
        print(f"Epoch {t}\n-------------------------------")

        train(train_dataloader, model, loss_fn, optimizer, False, False)

        percent_correct, test_loss = test(test_dataloader, model, loss_fn)
        acc.append(percent_correct)
        loss.append(test_loss)
    stop = timeit.default_timer()
    plt.close()
    fig = plt.figure()

    ax1 = plt.subplot(321)

    ax1.plot(indexed_epochs, acc)
    ax1.set_title("Stock, t: " + "{:.2f}".format(stop - start) + " s")
    ax1.set_xlabel("Epoch")
    ax1.set_xticks(indexed_epochs)
    ax1.set_yticks(np.arange(0, 110, 10))
    ax1.set_ylabel("% Accuracy")

    ax2 = plt.subplot(322)

    ax2.plot(indexed_epochs, loss)
    ax2.set_title("Stock, t: " + "{:.2f}".format(stop - start) + " s")
    ax2.set_xlabel("Epoch")
    ax2.set_xticks(indexed_epochs)
    ax2.set_ylabel("Loss")

    acc = []
    loss = []

    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=1e-2)

    start = timeit.default_timer()
    for t in indexed_epochs:
        print(f"Epoch {t}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, True, False)

        percent_correct, test_loss = test(test_dataloader, model, loss_fn)
        acc.append(percent_correct)
        loss.append(test_loss)
    stop = timeit.default_timer()

    ax3 = plt.subplot(323)

    ax3.plot(indexed_epochs, acc)
    ax3.set_title("rSVD, t: " + "{:.2f}".format(stop - start) + " s")
    ax3.set_xlabel("Epoch")
    ax3.set_xticks(indexed_epochs)
    ax3.set_yticks(np.arange(0, 110, 10))
    ax3.set_ylabel("% Accuracy")

    ax4 = plt.subplot(324)

    ax4.plot(indexed_epochs, loss)
    ax4.set_title("rSVD, t: " + "{:.2f}".format(stop - start) + " s")
    ax4.set_xlabel("Epoch")
    ax4.set_xticks(indexed_epochs)
    ax4.set_ylabel("% Accuracy")

    acc = []
    loss = []

    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=1e-2)

    start = timeit.default_timer()
    for t in indexed_epochs:
        print(f"Epoch {t}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, False, True)

        percent_correct, test_loss = test(test_dataloader, model, loss_fn)
        acc.append(percent_correct)
        loss.append(test_loss)
    stop = timeit.default_timer()

    ax5 = plt.subplot(325)

    ax5.plot(indexed_epochs, acc)
    ax5.set_title("truncSVD, t: " + "{:.2f}".format(stop - start) + " s")
    ax5.set_xlabel("Epoch")
    ax5.set_xticks(indexed_epochs)
    ax5.set_yticks(np.arange(0, 110, 10))
    ax5.set_ylabel("% Accuracy")

    ax6 = plt.subplot(326)

    ax6.plot(indexed_epochs, loss)
    ax6.set_title("truncSVD, t: " + "{:.2f}".format(stop - start) + " s")
    ax6.set_xlabel("Epoch")
    ax6.set_xticks(indexed_epochs)
    ax6.set_ylabel("% Accuracy")

    plt.tight_layout
    plt.show()


figure_1()
