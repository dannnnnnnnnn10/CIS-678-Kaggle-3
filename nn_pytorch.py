import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import csv
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(12345)

training_set_y = np.genfromtxt("mnist_train_targets.csv", delimiter=",", dtype=float, skip_header=1)[:]
training_set_x = np.genfromtxt("mnist_train.csv", delimiter=",", dtype=float, skip_header=1)[:,:]
test_set_x = np.genfromtxt("mnist_test.csv", delimiter=",", dtype=float, skip_header=1)[:,:]

X = training_set_x.T
test_X = test_set_x.T

for i in range(X[0].size):
    min = np.min(X[i])
    max = np.max(X[i])
    X[i] = (X[i] - min) / (max - min)

for i in range(test_X[0].size):
    min = np.min(test_X[i])
    max = np.max(test_X[i])
    test_X[i] = (test_X[i] - min) / (max - min)

x_train = torch.from_numpy(X)
x_train = x_train.to(torch.float32)
y_train = torch.from_numpy(training_set_y)
y_train = y_train.to(torch.long)
x_valid = torch.from_numpy(test_X)
x_valid = x_valid.to(torch.float32)

x_ds = TensorDataset(x_train, y_train)

train_ds, test_ds = random_split(x_ds, [0.7, 0.3])
train_dl = DataLoader(train_ds, batch_size=64)
test_dl = DataLoader(test_ds, batch_size=64)

device = "cpu"

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(784, 160),
            nn.ReLU(),
            nn.LayerNorm(160),
            nn.Linear(160, 45),
            nn.ReLU(),
            nn.Linear(45, 10),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        logits = self.stack(x)
        return logits

model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(str(100*correct))

epochs = 50
for t in range(epochs):
    # print(f"Epoch {t+1}\n-------------------------------")
    train(train_dl, model, loss_fn, optimizer)
    test(test_dl, model, loss_fn)
print("Done!")

with torch.no_grad():
    pred = model(x_valid)
    pred = pred.detach().numpy()

results = np.zeros(test_X[:,0].size)
for i in range(test_X[:,0].size):
    results[i] = np.argmax(pred[i])

with open("output.csv", 'w', newline='') as csvfile:
    outputwriter = csv.writer(csvfile, quotechar='"', delimiter=",")
    outputwriter.writerow(['Id'] + ['Expected'])
    for n in range(results.size):
        outputwriter.writerow([str(int(n+1))] + [str(int(results[n]))])