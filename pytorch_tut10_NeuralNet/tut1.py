import torch
import torch.nn as nn
import torch.nn.functional as F

# here nn.Linear(input_size, hidden_size) -> hidden_size = output_size / input size of next layer
# here in nn.Linear(hidden_size, 1) -> hidden_size = input_size of that layer, 1 = output size of that layer


# create (nn modules) -> option 1
class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)  # linear layer
        self.relu = nn.ReLU()  # relu activation function
        self.linear2 = nn.Linear(hidden_size, 1)  # linear layer
        self.sigmoid = nn.Sigmoid()   # sigmoid activation function

    def forward(self, x):
        out = self.linear1(x)  # input inserted into layer1
        out = self.relu(out)  # hidden layer 1
        out = self.linear2(out)  # hidden layer 2
        out = self.sigmoid(out)  # output layer
        return out


# create -> option 2

class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        return out


# So you have input features x. You feed them to the first layer, which is linear. The output size is reduced to the given hidden_size
# Then the output of that layer is passed through a ReLU function (layer 2). The output given by the ReLU is then passed through another linear layer (layer 3).
# The output size depends on the size specified. Here in the 3rd layer, it's 1. Now this output of this layer is passed through a sigmoid function (layer 4).
# This value is then checked with a threshold to predict the required answer.