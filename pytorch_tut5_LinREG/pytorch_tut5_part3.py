# Linear Regression using pytorch part 1

# 1) Design model (input , output, forward pass)
# 2) Construct Loss and Optimizer
# 3) Training Loop
#       - forward pass: compute prediction
#       - backward pass: computer gradients
#       - update weights

import torch
import torch.nn as nn

# Datasets -> X: Samples with features, Y: target/output of the samples
# 4 samples - 1 feature each, 1 target each
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features
# Above data tells us how the data should behave

model = nn.Linear(input_size, output_size)

# # Custom Linear Regression model
# class LinearRegression(nn.Module):
#
#     def __init__(self, input_dim, output_dim):
#         super(LinearRegression, self).__init__()
#         # define layers
#         self.lin = nn.Linear(input_dim, output_dim)
#
#     def forward(self, x):
#         return self.lin(x)
#
#
# model = LinearRegression(input_size, output_size)

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# Training
learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()  # loss
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):

    # prediction - forward pass
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    # gradient using backward propagation
    l.backward()  # dl/dw

    # update weights
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()  # sets the gradients of w to 0.

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch -> {epoch + 1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')

