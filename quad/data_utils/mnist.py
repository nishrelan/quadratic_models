from haiku import Flatten
import torch
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import jax
import jax.numpy as jnp
import sys


class NumpyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.data)

def collate_fn(samples):
    xb, yb = list(zip(*samples))
    xb = np.stack(xb)
    yb = np.array(yb)
    return xb, yb

class ToNumpyTransform:
    def __init__(self) -> None:
        pass
    def __call__(self, x):
        return x

class FlattenTransform:
    def __init__(self) -> None:
        pass
    def __call__(self, x):
        return x.reshape(-1)
  
class BinaryLabelTransform:
    def __init__(self):
        pass
    def __call__(self, y):
        if jnp.sum(y) % 2 == 0:
            return 1
        else:
            return 0

def get_mnist_binary(path, batch_size, train_size=None, test_size=None, rngs=None):
    """


    Params:
        batch_size: batch size for both train and test loaders
        train_size: number of data points to include in training set
        test_size: number of data points to include in test set
        rngs: rng keys needed to select random subsets of data of size train_size and test_size
    Returns:
        train and test data loaders 
    
    """
    train_ds = torchvision.datasets.MNIST(path, train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                ToNumpyTransform(),
                                FlattenTransform()
                                ]),
                                target_transform=torchvision.transforms.Compose([
                                    ToNumpyTransform(),
                                    BinaryLabelTransform()
                                ]))
    test_ds = torchvision.datasets.MNIST(path, train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                ToNumpyTransform(),
                                FlattenTransform()
                                ]),
                                target_transform=torchvision.transforms.Compose([
                                    ToNumpyTransform(),
                                    BinaryLabelTransform()
                                ]))

    # select random subset of train and test dataset   
    if train_size is not None:                
        key1, key2 = rngs

        train_perm = np.array(jax.random.permutation(key1, len(train_ds)))
        test_perm = np.array(jax.random.permutation(key2, len(test_ds)))

        train_ds = Subset(train_ds, train_perm[:train_size])
        test_ds = Subset(test_ds, test_perm[:test_size])

    trainloader = DataLoader(train_ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    testloader = DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    return trainloader, testloader