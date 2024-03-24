import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
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

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=64)

device = "cpu"

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(784, 250),
            nn.ReLU(),
            nn.Linear(250, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        logits = self.stack(x)
        return logits

model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

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
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dl, model, loss_fn, optimizer)
print("Done!")

pred = model(x_train)
pred = pred.detach().numpy()

correct = 0

for i in range(pred[:,0].size):
    if training_set_y[i] == np.argmax(pred[i]):
        correct += 1

print("Accuracy: " + str(correct / pred[:,0].size * 100))

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