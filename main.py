import os
import random
import argparse
import numpy as np
import torch
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision.transforms import Compose, ToPILImage, RandomCrop, RandomHorizontalFlip, ToTensor
from models import MLP, ResNet, DenseNet
from bias_correction.dataset import MixtureDataset
from bias_correction.experiment import Experiment
from bias_correction.utils import sample_n


parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=('mlp', 'resnet', 'densenet'), default='mlp')
parser.add_argument('--dataset', choices=('mnist', 'cifar10', 'cifar100'), default='mnist')
parser.add_argument('--entropy_weight', type=float, default=0.0)
parser.add_argument('--n_classes', type=int, default=5)
parser.add_argument('--n_valid', type=int, default=10000)
parser.add_argument('--bias', type=float, default=0.75)
parser.add_argument('--n_rounds', type=int, default=10)
parser.add_argument('--n_update', type=int, default=100)
parser.add_argument('--pseudo_labels', type=bool, default=False)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_step', type=int, action='append', default=[])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--log_dir', type=str, default='log')


def get_model(args):
    if args.model == 'mlp':
        model = MLP(num_classes=args.n_classes)
    elif args.model == 'resnet':
        model = ResNet(20, num_classes=args.n_classes)
    elif args.model == 'densenet':        
        model = DenseNet(40, num_classes=args.n_classes)        

    return model


def get_datasets(args):
    if args.dataset == 'mnist':
        biased_weights = [(args.bias, 1-args.bias) for _ in range(args.n_classes)]
        balanced_weights = [(1, 1) for _ in range(args.n_classes)]
        override = sample_n(range(10), [10//args.n_classes for _ in range(args.n_classes)])
        train_dataset = MNIST(root='./data/mnist', train=True, download=True)
        test_dataset = MNIST(root='./data/mnist', train=False, download=True)
        train_dataset.train_data = train_dataset.train_data.unsqueeze(-1).numpy()        
        test_dataset.test_data = test_dataset.test_data.unsqueeze(-1).numpy()
        transform = Compose([ToPILImage(),
                            RandomCrop(28, padding=4),                            
                            ToTensor()])

    elif args.dataset == 'cifar10':
        biased_weights = [(args.bias, 1-args.bias) for _ in range(args.n_classes)]
        balanced_weights = [(1, 1) for _ in range(args.n_classes)]
        override = sample_n(range(10), [10//args.n_classes for _ in range(args.n_classes)])
        train_dataset = CIFAR10(root='./data/cifar10', train=True, download=True)
        test_dataset = CIFAR10(root='./data/cifar10', train=False, download=True)     
        transform = Compose([ToPILImage(),
                             RandomCrop(32, padding=4),
                             RandomHorizontalFlip(),
                             ToTensor()])

    elif args.dataset == 'cifar100':  # uses predefined coarse labels
        coarse_labels = [(4, 30, 55, 72, 95), (1, 32, 67, 73, 91), (54, 62, 70, 82, 92), (9, 10, 16, 28, 61),
                         (0, 51, 53, 57, 83), (22, 39, 40, 86, 87), (5, 20, 25, 84, 94), (6, 7, 14, 18, 24), 
                         (3, 42, 43, 88, 97), (12, 17, 37, 68, 76), (23, 33, 49, 60, 71), (15, 19, 21, 31, 38), 
                         (34, 63, 64, 66, 75), (26, 45, 77, 79, 99), (2, 11, 35, 46, 98), (27, 29, 44, 78, 93), 
                         (36, 50, 65, 74, 80), (47, 52, 56, 59, 96), (8, 13, 48, 58, 90), (41, 69, 81, 85, 89)]
        biased_weights = [(args.bias, (1-args.bias)/4, (1-args.bias)/4, (1-args.bias)/4, (1-args.bias)/4)
                            for _ in range(args.n_classes)]
        balanced_weights = [(1, 1, 1, 1, 1) for _ in range(args.n_classes)]
        override = random.sample([random.sample(coarse_label, 5) for coarse_label in coarse_labels], args.n_classes)
        train_dataset = CIFAR100(root='./data/cifar100', train=True, download=True)
        test_dataset = CIFAR100(root='./data/cifar100', train=False, download=True)
        transform = Compose([ToPILImage(),
                             RandomCrop(32, padding=4),
                             RandomHorizontalFlip(),
                             ToTensor()])

    train_mixture = MixtureDataset(train_dataset.train_data[:-args.n_valid],
                                   train_dataset.train_labels[:-args.n_valid],
                                   mixture_weights=biased_weights,
                                   mixture_override=override,
                                   transform=transform)

    valid_mixture = MixtureDataset(train_dataset.train_data[-args.n_valid:],
                                   train_dataset.train_labels[-args.n_valid:],
                                   mixture_weights=balanced_weights,
                                   mixture_override=override,
                                   transform=transform)

    test_mixture = MixtureDataset(test_dataset.test_data,
                                  test_dataset.test_labels,
                                  mixture_weights=balanced_weights,
                                  mixture_override=override,
                                  transform=transform)

    return train_mixture, valid_mixture, test_mixture


if __name__ == '__main__':    
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = get_model(args)    
    train_dataset, valid_dataset, test_dataset = get_datasets(args)

    experiment = Experiment(model, train_dataset, valid_dataset, test_dataset, args)
    experiment.run()
