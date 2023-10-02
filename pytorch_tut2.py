import torch

x = torch.rand(3, requires_grad=True)  # flag that controls whether a tensor requires a gradient or not.
# if true starts tracking all the operation history and forms a backward graph for gradient calculation.
print(x)

y = x+2
print(y)

z = y*y*2

# z = z.mean()
print(z)

# z.backward()  # calc the gradient dz/dx
# print(x.grad)

v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
z.backward(v)  # dz/dx
print(x.grad)


# prevent from tracking the event for gradient. Op not a part of the calculation
# x.requires_grad_(False)
# x.detach()
# with torch.no_grad()

# x.requires_grad_(False)
# print(x)

# y = x.detach()
# print(y)

# with torch.no_grad():
#     y = x + 2
#     print(y)