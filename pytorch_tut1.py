import torch
import numpy as np

x = torch.empty(2,2,3) # empty tensor

y = torch.rand(2, 2) # random tensor

z = torch.zeros(3, 4) # All zeros array

a = torch.ones(2, 3, dtype=torch.int16) # All ones

# Can also set the datatypes while forming the arrays --> .dtype
# print(a.dtype)

# BASIC OPS

xy = torch.tensor([[3.4, 5.6], [4.5,6.7]])

x1 = torch.rand(2, 2)
y1 = torch.rand(2, 2)
print(x1, y1)

# Add tensors
z1 = x1 + y1  # or torch.add(x1, y1)
print('sum: ', z1)

# Sub tensors
z2 = y1 - x1  # or torch.sub(y1, x1)
print('sub: ', z2)

# Multiply tensors
z3 = x1 * y1  # or torch.mul(x1, y1)
print('mult: ', z3)


y1.add_(x1)  # adds x1 to y1 and stores in y1
print('y1: ', y1)

# SLICING AND ACCESSING

a1 = torch.rand(5, 3)
print(a1)
print(a1[1, 1].item())  # to get that specific value

b1 = a1.view(15)  # converted a1 tensor into a 1D array of size = #elements in a1
print(b1)

c1 = a1.view(-1, 5)  # -1 means you are letting system decide that value
print(c1)

b2 = a1.numpy()  # create numpy array from tensor
print(b2)

# Let's change a1 and see if it changes b2 too
a1.add_(5)
print(a1)
print(b2)  # yes it changed b2 too

# create numpy array, convert to tensor from numpy and store in another
a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)

# let's again check the same for this too
a += 1
print(a)
print(b)  # yes, on changing a, b got changed too

