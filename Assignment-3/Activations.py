import numpy as np

# activation function and its derivative
def tanh(x):
    return np.tanh(x);

def tanh_prime(x):
    return 1-np.tanh(x)**2;

def sigmoid(x):
    return 1/(1 + np.exp(-x));

def sigmoid_prime(x):
    return (1/(1 + np.exp(-x))) * (np.exp(-x)/(1 + np.exp(-x)));