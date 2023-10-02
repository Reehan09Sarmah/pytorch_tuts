# Cross Entropy
import torch
import torch.nn as nn

# cross entropy using pytorch
# cross entropy of pytorch automatically uses the softmax on the inputs,
# so we don't need to use the softmax separately in last layer
print('Using cross entropy of pytorch')
loss = nn.CrossEntropyLoss()

# 3 samples = each sample tells which index it belongs to
# for eg: 2 means index 2 should have higher value
# each index number represents a particular class
Y = torch.tensor([2, 0, 1])

# nsamples x nclasses = 1 x 3
Y_pred_good = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [1.0, 3.0, 0.1]])
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3], [0.2, 1.0, 3.3], [2.5, 0.2, 0.3]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)
print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2:.4f}')

_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
# which value did it predict
print('predictions:')
print(f'According to Y_pred_good: {predictions1}')
print(f'According to Y_pred_bad: {predictions2}')
