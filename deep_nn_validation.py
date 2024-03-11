import numpy as np
import csv
from sklearn.neural_network import MLPRegressor as nn

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

train, test = train_test_split(A, random_state=12345)

test_X = test[:,25:]
test_Y = test[:,0:25]
train_X = train[:,25:]
train_Y = train[:,0:25]

coefficients = np.empty(10)

for k in range(10):
    clf = nn(random_state=12345, max_iter=1000,  learning_rate_init=0.0001, hidden_layer_sizes=(225,100), activation='logistic', beta_1=(0.98), beta_2=(0.999-k*0.001)).fit(train_X, train_Y)
    results = clf.predict(test_X)

    coefficients[k] = np.corrcoef(np.matrix.flatten(results), np.matrix.flatten(test_Y))[0,1]
    print('Run done. Coef at k=' + str(0.999-k*0.001) + ': ' + str(coefficients[k]))

with open("validation.csv", 'w', newline='') as csvfile:
    outputwriter = csv.writer(csvfile, quotechar='"', delimiter=",")
    outputwriter.writerow(['Alpha'] + ['Coefficient'])
    for n in range(coefficients.size):
        outputwriter.writerow([str(int((n+1)*0.0001))] + [str(float(coefficients[n]))])