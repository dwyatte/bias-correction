import itertools
import numpy as np
from collections import OrderedDict
from torch.utils.data.dataset import Dataset
from bias_correction.utils import sample_n


class MixtureDataset(Dataset):

    def __init__(self, X, y,
                mixture_weights=None, mixture_override=None,
                transform=None):

        if mixture_weights is None:
            mixture_weights = [(1,) for _ in set(y)]

        new_classes = range(len(mixture_weights))

        if mixture_override is None:
            mixture = sample_n(set(y), list(map(len, mixture_weights)))
        else:
            mixture = mixture_override

        forward_map = OrderedDict()
        backward_map = OrderedDict(zip(new_classes, mixture))
        indices = OrderedDict(zip(new_classes, ([] for _ in new_classes)))

        for new_class, old_classes, weights in zip(new_classes, mixture, mixture_weights):
            for old_class, weight in zip(old_classes, weights):
                index = np.where(np.asarray(y) == old_class)[0]
                n = int(len(index) * weight)
                indices[new_class] += list(np.random.permutation(index)[:n])
                forward_map[old_class] = new_class

        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.mixture_weights = mixture_weights
        self.forward_map = forward_map
        self.backward_map = backward_map
        self.indices = indices
        self.transform = transform

    def __getitem__(self, index):
        offsets = np.cumsum(list(map(len, self.indices.values())))
        y = np.where(index < offsets)[0][0]

        if y == 0:
            index = self.indices[y][index]
        else:
            index = self.indices[y][index-offsets[y-1]]

        x = self.X[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return np.sum(list(map(len, self.indices.values())))

    def get(self, indices):
        offsets = np.cumsum(list(map(len, self.indices.values())))

        xs = np.zeros((len(indices),)+self.X.shape[1:], dtype=self.X.dtype)
        ys = np.zeros(len(indices), dtype=self.y.dtype)

        for i, index in enumerate(indices):
            y = np.where(index < offsets)[0][0]

            if y == 0:
                index = self.indices[y][index]
            else:
                index = self.indices[y][index-offsets[y-1]]             

            xs[i] = self.X[index]
            ys[i] = self.y[index]

        return xs, ys

    def add(self, xs, ys):
        for i, (x, y) in enumerate(zip(xs, ys)):
            self.X = np.append(self.X, xs[i:i+1], axis=0)
            self.y = np.append(self.y, ys[i:i+1], axis=0)
            self.indices[self.forward_map[y]].append(len(self.y)-1)

    def drop(self, indices):
        for _class in indices:
            self.indices[_class] = list(set(self.indices[_class])-set(indices[_class]))

    def effective_mixture_weights(self):
        effective_mixture_weights = []

        for new_class in self.backward_map:
            weights = []
            for old_class in self.backward_map[new_class]:
                old_classes = np.asarray(self.y)[self.indices[new_class]] 
                weight = sum(old_classes == old_class)/len(self.indices[new_class])
                weights.append(weight)
            effective_mixture_weights.append(weights)
        
        return effective_mixture_weights
