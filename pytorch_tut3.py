import torch


weights = torch.ones(4, requires_grad=True)

weights.grad.zero_()  # to empty the gradients before moving onto the next

# for epoch in range(3):
#     model_output = (weights*3).sum()
#
#     model_output.backward()
#
#     print(weights.grad)
#
#     weights.grad.zero_()
