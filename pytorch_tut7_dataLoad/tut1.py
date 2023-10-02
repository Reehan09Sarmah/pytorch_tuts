# How to load data from csv files and access them batch wise
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import multiprocessing

class WineDataset(Dataset):

    def __init__(self):
        # data loading
        xy = np.loadtxt('wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])  # n_samples, 1
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


# Do this in Windows machine so that you can use num_workers for multiprocessing.
if __name__ == '__main__':
    # This line is required to ensure proper process spawning on Windows
    multiprocessing.freeze_support()

    dataset = WineDataset()
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)

    num_epochs = 2
    total_samples = len(dataset)
    n_iterations = math.ceil(total_samples / 4)

    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(dataloader):
            if (i + 1) % 5 == 0:
                print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_iterations}, inputs {inputs.shape}')






