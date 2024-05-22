"""
    Created on Tue 27 Oct 2020

    @author: umairkarel
"""

import numpy as np
import pickle

# Classes
class NeuralNetwork:
    def __init__(self, numI, numH, numO):
        self.input_nodes = numI
        self.hidden_nodes = numH
        self.output_nodes = numO
        self.alpha = 0.1

        self.weigths_ih = np.random.uniform(-1, 1, (self.hidden_nodes, self.input_nodes))
        self.weigths_ho = np.random.uniform(-1, 1, (self.output_nodes, self.hidden_nodes))
        self.bias_h = np.random.uniform(-1, 1, (self.hidden_nodes, 1))
        self.bias_o = np.random.uniform(-1, 1, (self.output_nodes, 1))

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def drev_sigmoid(self, y):
        return np.multiply(y,(1-y))

    def feedforward(self, inputs):
        hidden = (self.weigths_ih @ inputs).reshape(-1,1)
        hidden = np.add(hidden, self.bias_h)
        hidden = self.sigmoid(hidden)

        output = (self.weigths_ho @ hidden).reshape(-1,1)
        output = np.add(output, self.bias_o)
        output = self.sigmoid(output)

        return output.flatten()

    def train(self, inputs, targets):
        # FeedForward
        hidden = (self.weigths_ih @ inputs).reshape(-1,1)
        hidden = np.add(hidden, self.bias_h)
        hidden = self.sigmoid(hidden)

        output = (self.weigths_ho @ hidden).reshape(-1,1)
        output = np.add(output, self.bias_o)
        output = self.sigmoid(output)

        # Calculating Errors
        targets = targets.reshape(-1,1)
        output_error = np.subtract(targets, output)
        hidden_errors = self.weigths_ho.T @ output_error 

        # BackPropagation
        gradient = self.drev_sigmoid(output)
        gradient = np.multiply(gradient, output_error)
        gradient *= self.alpha

        weigths_ho_delta = gradient @ hidden.T

        self.weigths_ho = np.add(self.weigths_ho, weigths_ho_delta)
        self.bias_o = np.add(self.bias_o, gradient)

        hidden_grad = self.drev_sigmoid(hidden)
        hidden_grad = np.multiply(hidden_grad, hidden_errors)
        hidden_grad *= self.alpha
        weigths_ih_delta = hidden_grad @ inputs.reshape(1,-1)  

        self.weigths_ih = np.add(self.weigths_ih, weigths_ih_delta)
        self.bias_h = np.add(self.bias_h, hidden_grad)



# Using Trained Model
# nn = pickle.load(open('xor_model.pickle', 'rb'))
model = NeuralNetwork(3, 10, 1)

# XOR Table
training_data = np.array([[1,1,1,1],
                          [1,1,0,0],
                          [1,0,1,0],
                          [1,0,0,1],
                          [0,1,0,1],
                          [0,0,0,0]])

for _ in range(2000):
    np.random.shuffle(training_data)
    for i in range(len(training_data)):
        model.train(training_data[i, :3], training_data[i, -1])

print(model.feedforward([0,1,1]), model.feedforward([0,0,1]))

# Saving Trained Model
# with open('xor_model.pickle', 'wb') as f:
# 	pickle.dump(nn, f)
# pickle_in = open('xor_model.pickle', 'rb')