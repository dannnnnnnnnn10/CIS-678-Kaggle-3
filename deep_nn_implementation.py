import numpy as np
import csv
import matplotlib.pyplot as plt

training_set_y = np.genfromtxt("mnist_train_targets.csv", delimiter=",", dtype=float, skip_header=1)[:]
training_set_x = np.genfromtxt("mnist_train.csv", delimiter=",", dtype=float, skip_header=1)[:,:]
test_set_x = np.genfromtxt("mnist_test.csv", delimiter=",", dtype=float, skip_header=1)[:,:]

Y = np.zeros((training_set_y.size, 10))
for i in range(training_set_y.size):
    num = int(training_set_y[i])
    Y[i, num] = 1
X = training_set_x.T
test_X = test_set_x.T

print(test_X[0].size)
print(test_X[:,0].size)

for i in range(X[0].size):
    min = np.min(X[i])
    max = np.max(X[i])
    X[i] = (X[i] - min) / (max - min)

for i in range(test_X[0].size):
    min = np.min(test_X[i])
    max = np.max(test_X[i])
    test_X[i] = (test_X[i] - min) / (max - min)

# X = np.log1p(X)
# test_X = np.log1p(test_X)

# plt.imshow(X[0].reshape(28, 28), cmap='gray')
# plt.show()

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

def softmax(y):
    return np.exp(y) / sum(np.exp(y))

def d_softmax(z):
    gz = softmax(z)
    side_length = z.size
    deriv = np.zeros((side_length, side_length))
    for i in range(side_length):
        for j in range(side_length):
            if (i == j):
                deriv[i,j] = gz[i] * (1 - gz[i])
            else:
                deriv[i,j] = -gz[i] * gz[j]
    result = np.zeros(side_length)
    for i in range(side_length):
        result[i] = np.sum(deriv[i])
    return result

def sigmoid(y):
    return 1 / (1 + np.exp(-y))

def d_sigmoid(z):
    return z * (1 - z)

def forward(x, w, b, funct):
    y = np.dot(x, w) + b
    if funct == "ReLU":
        return ReLU(y)
    elif funct == "Sigmoid":
        return sigmoid(y)
    elif funct == "Softmax":
        return softmax(y)
    else:
        return y

def glorot_uniform(size_in, size_out, random_state=12345):
    rng = np.random.default_rng(random_state)
    limit = np.sqrt(6. / (size_in + size_out))
    return rng.uniform(low=-limit, high=limit, size=(size_in, size_out)).astype('float32')

def he_init(size_in, size_out, random_state=12345):
    rng = np.random.default_rng(random_state)
    std_dev = np.sqrt(2 / size_in)
    weights = rng.normal(loc=0.0, scale=std_dev, size=(size_in, size_out))
    return weights

def binary_cross_entropy(y_true, y_pred):
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

def deep_nn(X, y, hidden_size_1, hidden_size_2, num_epochs=100, learning_rate=0.001, random_state=12345) :
    if random_state is not None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()
    w1 = he_init(X[0].size, hidden_size_1, random_state)
    w2 = he_init(hidden_size_1, hidden_size_2, random_state)
    # w3 = he_init(hidden_size_2, hidden_size_3, random_state)
    wn = he_init(hidden_size_2, 10, random_state)
    b1 = np.zeros(hidden_size_1)
    b2 = np.zeros(hidden_size_2)
    # b3 = np.zeros(hidden_size_3)
    bn = np.zeros(y[0].size)
    stochastic_order = np.arange(0, X.T[0].size)
    for epoch in range(num_epochs):
        correct_count = 0
        rng.shuffle(stochastic_order)
        for i in stochastic_order:

            z1 = forward(X[i,], w1, b1, "ReLU")
            z2 = forward(z1, w2, b2, "ReLU")
            # z3 = forward(z2, w3, b3, "ReLU")
            output = forward(z2, wn, bn, "Softmax")

            if np.argmax(output) == np.argmax(y[i]):
                correct_count += 1
            error = y[i] - output
            dyn = error * d_softmax(output)
            # dy3 = np.dot(dyn, wn.T) * dReLU(z3)
            dy2 = np.dot(dyn, wn.T) * dReLU(z2)
            dy1 = np.dot(dy2, w2.T) * dReLU(z1)
            
            wn = wn + (learning_rate * np.outer(z2, dyn))
            bn = bn + (learning_rate * dyn)
            # w3 = w3 + (learning_rate * np.outer(z2, dy3))
            # b3 = b3 + (learning_rate * dy3)
            w2 = w2 + (learning_rate * np.outer(z1, dy2))
            b2 = b2 + (learning_rate * dy2)
            w1 = w1 + (learning_rate * np.outer(X[i,], dy1))
            b1 = b1 + (learning_rate * dy1)
        pct_accurate = round(correct_count/stochastic_order.size * 100, 2)
        print('Epoch ' + str(epoch) + ' done. Accuracy = ' + str(pct_accurate) + "%.")
    return w1, w2, wn, b1, b2, bn

# train, test = train_test_split(A, random_state=12345)

# test_X = test[:,25:]
# test_Y = test[:,0:25]
# train_X = train[:,25:]
# train_Y = train[:,0:25]

w1, w2, wn, b1, b2, bn = deep_nn(X, Y, 150, 50, num_epochs=10)

z1 = forward(test_X, w1, b1, "ReLU")
z2 = forward(z1, w2, b2, "ReLU")
# z3 = forward(z2, w3, b3, "ReLU")
probs = forward(z2, wn, bn, "Softmax")

results = np.zeros(test_X[:,0].size)
for i in range(test_X[:,0].size):
    results[i] = np.argmax(probs[i])

with open("output.csv", 'w', newline='') as csvfile:
    outputwriter = csv.writer(csvfile, quotechar='"', delimiter=",")
    outputwriter.writerow(['Id'] + ['Expected'])
    for n in range(results.size):
        outputwriter.writerow([str(int(n+1))] + [str(int(results[n]))])