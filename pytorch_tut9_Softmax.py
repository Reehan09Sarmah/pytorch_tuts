# Softmax
import torch
import numpy as np

# Softmax->takes an array of inputs, converts into an array of probabilities
# all the probabilities sum up to 1. Used in output layer in Neural Networks


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# Using the softmax of our own
print('Using softmax of our own')
x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print('softmax numpy: ', outputs)

# using softmax provided by pytorch
print('Using softmax of our pytorch')
x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)
print('softmax tensor: ', outputs)
