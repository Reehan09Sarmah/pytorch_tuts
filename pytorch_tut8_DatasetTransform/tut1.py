import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np


# load the wine dataset as an object
class WineDataset(Dataset):

    def __init__(self, transform=None):
        # data loading
        xy = np.loadtxt('wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        # we don't convert to tensors here
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples


# 1) Our Own Transform
class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)


# 2) Our Own Transform
class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self,sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets


dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))

# Using the above 2 defined transforms we can create a composed transform
composed_transform = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataset(transform=composed_transform)
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))