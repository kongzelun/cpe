import torch
from torch import tensor
from torch.utils.data import Dataset
from pandas import read_csv
from random import sample

DATASETS = ['fm', 'c10']


class FashionMnist(Dataset):
    tensor_view = (1, 28, 28)
    train_test_split = 6000
    path = 'dataset/fashion-mnist_stream.csv'

    def __init__(self, train=True):
        dataset = read_csv(self.path, sep=',', header=None).values

        if train:
            dataset = dataset[:self.train_test_split]
        else:
            dataset = dataset[self.train_test_split:]

        self.data = []
        self.train = train
        self.label_set = set(dataset[:, -1].astype(int))

        for s in dataset:
            x = (tensor(s[:-1], dtype=torch.float) / 255).view(self.tensor_view)
            y = tensor(s[-1], dtype=torch.long)
            self.data.append((x, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class Cifar10(Dataset):
    tensor_view = (3, 32, 32)
    train_test_split = 6000
    path = 'dataset/cifar10_stream.csv'

    def __init__(self, train=True):
        dataset = read_csv(self.path, sep=',', header=None).values

        if train:
            dataset = dataset[:self.train_test_split]
        else:
            dataset = dataset[self.train_test_split:]

        self.data = []
        self.train = train
        self.label_set = set(dataset[:, -1].astype(int))

        for s in dataset:
            x = (tensor(s[:-1], dtype=torch.float) / 255).view(self.tensor_view)
            y = tensor(s[-1], dtype=torch.long)
            self.data.append((x, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class NoveltyDataset(Dataset):
    def __init__(self, dataset):
        self.data = sample(dataset.data, 2000)
        self.label_set = set(dataset.label_set)

    def extend(self, buffer, percent):
        assert 0 < percent <= 1
        self.data.extend(sample(buffer, int(percent * len(buffer))))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
