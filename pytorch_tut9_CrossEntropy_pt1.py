# Cross Entropy
import torch
import torch.nn as nn
import numpy as np

# Cross Entropy is the loss function.


def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss


print('Using cross entropy of our own')
Y = np.array([1, 0, 0])

# y_pred has probabilities. values after using softmax
Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2:.4f}')

# cross entropy using pytorch
# cross entropy of pytorch automatically uses the softmax on the inputs,
# so we don't need to use the softmax separately in last layer
print('Using cross entropy of pytorch')
loss = nn.CrossEntropyLoss()

Y = torch.tensor([0])  # 1 sample --> which class it belongs to
# which index should have high probability
# each index number represents a particular class

# nsamples x nclasses = 1 x 3
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)
print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2:.4f}')

_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(predictions1)
print(predictions2)
