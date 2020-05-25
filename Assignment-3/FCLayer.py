import numpy as np

# inherit from base class Layer
class FCLayer:
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size, weights = "normal"):
        if weights == "0":
            print(input_size)
            self.weights = np.zeros(shape = (input_size, output_size))
            self.bias = np.zeros(shape = (1, output_size))
        elif weights == "gaussian":
            self.weights = np.random.randn(input_size, output_size)
            self.bias = np.random.randn(1, output_size)
        elif weights == "uniform":
            self.weights = np.random.uniform(size = (input_size, output_size))
            self.bias = np.random.uniform(size = (1, output_size))
        elif weights == "xavier":
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(1 / input_size)
            self.bias = np.random.randn(1, output_size) * np.sqrt(1 / input_size)
        elif weights == "normal":
            self.weights = np.random.rand(input_size, output_size) - 0.5
            self.bias = np.random.rand(1, output_size) - 0.5

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error