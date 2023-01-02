# This is a sample code of neural network with one hidden layer, written from scratch by Ozan KAYA.
# https://github.com/kayaozan

# The dataset is called Titanic.
# It contains several data of passengers such as sex, age, fare, and whether they survived or not.
# The dataset can be obtained from https://www.openml.org/search?type=data&sort=runs&id=40945

# The aim of this code is to estimate the parameters of logistic model
# and predict the survival outcome.

def load_and_prepare():
# Loads the data, divides it into train and test sets and returns them.

    data = pd.read_csv('titanic3.csv',index_col=False)

    # Replacing text variables with binary values
    data.replace({'male': 1, 'female': 0}, inplace=True)
    
    # Dropping the rows that contain NaN and selecting the columns that will be used
    data = data[['sex', 'pclass','age','fare','survived']].dropna().to_numpy()    
    
    # 20% of the data to be selected for test purposes
    test_size = 0.2
    test_range = int(data.shape[0] * test_size)

    # Shuffling the data
    np.random.seed(0)
    np.random.shuffle(data)

    # Dividing the data set into training and test sets
    # Reshaping the arrays such that they work well with matrix operations
    x_test  = data[:test_range, 1:].reshape(test_range, -1)
    x_train = data[test_range:, 1:].reshape(data.shape[0]-test_range, -1)
    
    y_test  = data[:test_range, 0].reshape(-1, 1)
    y_train = data[test_range:, 0].reshape(-1, 1)

    return x_train, y_train, x_test, y_test

def normalize(x_train, x_test):
# Normalizes the feature sets.
# Uses the mean and standard deviation of the training set for both
# so that the regression model is not biased.

    trainMean = np.mean(x_train, axis=0)
    trainStd  = np.std( x_train, axis=0)
    
    x_train = (x_train - trainMean) / trainStd
    x_test  = (x_test  - trainMean) / trainStd

    return x_train, x_test

def sigmoid(z):
# Sigmoid function.

    return 1 / (1 + np.exp(-z))

def initialize_parameters(x, y, size_hl):

    size_x = x.shape[1]
    size_y = y.shape[1]

    # Weights and biases of both layers are generated seperately.
    w1 = 0.01 * np.random.randn(size_x, size_hl)
    b1 = np.zeros([1, size_hl])
    w2 = 0.01 * np.random.randn(size_hl, size_y)
    b2 = np.zeros([1, size_y])

    # Parameters are stored in a dictionary to be sent.
    parameters = {'w1': w1,
                  'b1': b1,
                  'w2': w2,
                  'b2': b2}

    return parameters

def forward_propagate(x, parameters):
# Forward propagation step of the neural network.

    # Unfolding the parameters dictionary
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    # Calculating the activations:
    # tanh is used for the hidden layer,
    # and sigmoid is used for the output layer.
    z1 = x @ w1 + b1
    a1 = np.tanh(z1)
    z2 = a1 @ w2 + b2
    a2 = sigmoid(z2)

    cache = {'z1': z1,
             'a1': a1,
             'z2': z2,
             'a2': a2}

    return a2, cache

def cost_function(a2, y):
# Calculates the error for each sample.
# Uses the formula: (-1/m) * sum( y.T log(a2) + (1-y) log(1-a2) )
# where m is the size of samples

    cost = -1/y.shape[0] * (y.T @ np.log(a2) + (1 - y).T @ np.log(1 - a2))
    
    return np.squeeze(cost)

def back_propagate(parameters, cache, x, y, learning_rate = 1.0):
# Back propagation step of the neural network.

    m = x.shape[0]
    
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    a1 = cache["a1"]
    a2 = cache["a2"]

    # Gradient calculations
    dz2 = a2 - y
    dw2 = (1/m)* a1.T @ dz2
    db2 = (1/m) * np.sum(dz2,axis=0)
    dz1 = dz2 @ w2.T *(1 - a1** 2)
    dw1 = (1/m)* x.T @ dz1
    db1 = (1/m) * np.sum(dz1,axis=0)

    # Updating the parameters
    w2 = w2 - learning_rate * dw2
    b2 = b2 - learning_rate * db2
    w1 = w1 - learning_rate * dw1
    b1 = b1 - learning_rate * db1

    parameters = {'w1': w1,
                  'b1': b1,
                  'w2': w2,
                  'b2': b2}

    return parameters

def NN(x, y, size_hl = 3, epocs = 1000):
# Neural network with one hidden layer.
# size_hl is the size of hidden layer.

    parameters = initialize_parameters(x, y, size_hl)

    cost = []
    for i in range(epocs):
        
        # Forward propagation
        a2, cache = forward_propagate(x, parameters)
        
        # Computation of cost
        cost.append(cost_function(a2, y))

        # Back propagation and updating the parameters
        parameters = back_propagate(parameters, cache, x, y)

    return parameters, cost

def predict(x, parameters):
# Predicts the output values for given features and parameters.
# The prediction is 1 if the calculation is equal or greater than 0.5,
# or 0 if it is less than 0.5

    # Calculating the activation for test set
    a2, _ = forward_propagate(x, parameters)

    return a2 >= 0.5

def score(y_prediction, y):
# Calculates the ratio of true predictions to all.

    return np.sum(y_prediction == y) / y.shape[0]

import pandas as pd
import numpy as np
import matplotlib.pyplot as pl

x_train, y_train, x_test, y_test = load_and_prepare()

# Normalizing the feature sets
x_train, x_test = normalize(x_train, x_test)

# Sending the training set to neural network
parameters, cost = NN(x_train, y_train)

# Plotting the cost for monitoring
pl.plot(cost)

# Testing the model with the test set
y_test_prediction = predict(x_test, parameters)
accuracy = score(y_test_prediction, y_test)
print('The accuracy measured with test set:\n', accuracy)