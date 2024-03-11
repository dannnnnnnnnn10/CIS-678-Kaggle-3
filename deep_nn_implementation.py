import numpy as np
import csv

training_set_y = np.genfromtxt("training_set_adt.csv", delimiter=",", dtype=float, skip_header=1)[:,1:]
training_set_x = np.genfromtxt("training_set_rna.csv", delimiter=",", dtype=float, skip_header=1)[:,1:]
test_set_x = np.genfromtxt("test_set_rna.csv", delimiter=",", dtype=float, skip_header=1)[:,1:]

X = training_set_x.T
Y = training_set_y.T
test_X = test_set_x.T

A = np.concatenate((Y, X), axis=1)

def train_test_split(X, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    np.random.shuffle(X)
    split_point = int(test_size * len(X))

    train, test = X[split_point:], X[:split_point]
    return train, test

def ReLU (y):
    z = np.where(y<0, 0, y)
    return z

def dReLU (z):
    y = np.where(z<=0, 0, 1)
    return y

def sigmoid(y):
    return 1 / (1 + np.exp(-y))

def d_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

def forward(x, w, b, funct):
    y = np.dot(x, w) + b
    if funct == "ReLU":
        return ReLU(y)
    elif funct == "Sigmoid":
        return sigmoid(y)
    else:
        return y
    
def backward(dz, z, zi, w, funct):
    if funct == "ReLU":
        dy = dz * dReLU(z)
    elif funct == "Sigmoid":
        dy = dz * d_sigmoid(z)
    else:
        dy = dz * z
    gradient = np.outer(zi, dy)
    dz2 = np.dot(dy, w.T)
    return gradient, dy, dz2

def deep_nn(X, y, hidden_size_1, hidden_size_2, num_epochs=100, learning_rate=0.0001, random_state=12345, funct=None) :
    if random_state is not None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()
    w1 = rng.normal(scale=1/X[0].size, size=(X[0].size, hidden_size_1))
    w2 = rng.normal(scale=1/np.sqrt(hidden_size_1), size=(hidden_size_1, hidden_size_2))
    wn = rng.normal(scale=1/np.sqrt(hidden_size_2), size=(hidden_size_2, y[0].size))
    b1 = np.zeros(hidden_size_1)
    b2 = np.zeros(hidden_size_2)
    bn = np.zeros(y[0].size)
    stochastic_order = np.arange(0, X.T[0].size)
    for epoch in range(num_epochs):
        rng.shuffle(stochastic_order)
        for i in stochastic_order:

            z1 = forward(X[i,], w1, b1, funct)
            z2 = forward(z1, w2, b2, funct)
            output = forward(z2, wn, bn, funct)
        
            error = y[i] - output
            gradientn, dyn, dz2 = backward(error, output, z2, wn, funct)
            gradient2, dy2, dz1 = backward(dz2, z2, z1, w2, funct)
            gradient1, dy1  = backward(dz1, z1, X[i,], w1, funct)[0:2]
            
            
            wn = wn + (learning_rate * (gradientn))
            bn = bn + (learning_rate * dyn)
            w2 = w2 + (learning_rate * gradient2)
            b2 = b2 + (learning_rate * dy2)
            w1 = w1 + (learning_rate * gradient1)
            b1 = b1 + (learning_rate * dy1)
        print('Epoch ' + str(epoch) + ' done.')
    return w1, w2, wn, b1, b2, bn

train, test = train_test_split(A, random_state=12345)

# test_X = test[:,25:]
# test_Y = test[:,0:25]
# train_X = train[:,25:]
# train_Y = train[:,0:25]

w1, w2, wn, b1, b2, bn = deep_nn(X, Y, 100, 100, funct="ReLU")

z1 = forward(test_X, w1, b1, "ReLU")
z2 = forward(z1, w2, b2, "ReLU")
results = forward(z2, wn, bn, "ReLU")

results = results.reshape((-1, 1))

with open("output.csv", 'w', newline='') as csvfile:
    outputwriter = csv.writer(csvfile, quotechar='"', delimiter=",")
    outputwriter.writerow(['Id'] + ['Expected'])
    for n in range(results.size):
        outputwriter.writerow(['ID_' + str(int(n+1))] + [str(float(results[n][0]))])