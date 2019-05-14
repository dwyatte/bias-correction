import os
import csv
import pickle
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchnet.meter import AverageValueMeter
from bias_correction.utils import merge_dictionaries


def entropy_regularized_loss(probs, targets, entropy_weight=0.0):
    probs = torch.clamp(probs, 1e-7, 1-1e-7)
    log_probs = torch.log(probs)
    nll = F.nll_loss(log_probs, targets, reduction='none')
    entropy = -torch.sum(probs*log_probs, dim=1)
    loss = torch.mean(nll-entropy_weight*entropy)

    return loss


def top1_accuracy(probs, targets):
    _, indices = torch.topk(probs, 1, dim=1)
    indices = indices.squeeze()
    corrects = torch.eq(indices, targets).float()
    accuracy = torch.mean(corrects)

    return accuracy


def weight_reset(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        module.reset_parameters()


class Experiment:

    def __init__(self, model, train_dataset, valid_dataset, test_dataset, args):        
        self.model = model        
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.entropy_weight = args.entropy_weight
        self.n_rounds = args.n_rounds
        self.n_update = args.n_update        
        self.pseudo_labels = args.pseudo_labels
        self.n_epochs = args.n_epochs
        self.lr = args.lr
        self.lr_step = args.lr_step
        self.batch_size = args.batch_size        
        self.log_dir = args.log_dir
        
        self.epoch = 0
        self.round = 0        
        self.meters = {meter: AverageValueMeter() for meter in
                        ('train_loss', 'train_accuracy', 'test_loss', 'test_accuracy')}

        os.makedirs(self.log_dir)
        
    def environment_save(self):
        torch.save(self.model, os.path.join(self.log_dir, 'model.pt'))

        with open(os.path.join(self.log_dir, 'train_dataset.pkl'), 'wb') as f:
            pickle.dump(self.train_dataset, f)

        with open(os.path.join(self.log_dir, 'test_dataset.pkl'), 'wb') as f:
            pickle.dump(self.test_dataset, f)

    def epoch_save(self):
        with open(os.path.join(self.log_dir, 'epoch_metrics.tsv'), 'a') as f:
            metrics = {meter: self.meters[meter].value()[0] for meter in self.meters}
            other = {'epoch': self.epoch, 'round': self.round}            
            dicts = merge_dictionaries((metrics, other))

            writer = csv.DictWriter(f, delimiter='\t', fieldnames=dicts.keys())
            if f.tell() == 0:
                writer.writeheader()

            writer.writerow(dicts)

    def round_save(self):
        with open(os.path.join(self.log_dir, 'round_metrics.tsv'), 'a') as f:
            metrics = {meter: self.meters[meter].value()[0] for meter in self.meters}            
            other = {'epoch': self.epoch, 'round': self.round, 
                     'train_bias': self.train_dataset.effective_mixture_weights()}
            dicts = merge_dictionaries((metrics, other))

            writer = csv.DictWriter(f, delimiter='\t', fieldnames=dicts.keys())
            if f.tell() == 0:
                writer.writeheader()

            writer.writerow(dicts)

    def train(self):        
        self.epoch = 0
        self.model.apply(weight_reset)
        
        optimizer = Adam(self.model.parameters(), lr=self.lr)            

        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                      shuffle=True, num_workers=4, pin_memory=True)
        test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size,
                                     shuffle=False, num_workers=4, pin_memory=True)

        for epoch in range(self.n_epochs):
                        
            for meter in self.meters:
                self.meters[meter].reset()

            self.model.train()
            for inputs, labels in train_dataloader:
                inputs = inputs.cuda()
                labels = labels.cuda()

                outputs = self.model(inputs)
                loss = entropy_regularized_loss(outputs, labels, self.entropy_weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                accuracy = top1_accuracy(outputs, labels)

                self.meters['train_loss'].add(loss.item())
                self.meters['train_accuracy'].add(accuracy.item())
            
            self.model.eval()
            for inputs, labels in test_dataloader:
                inputs = inputs.cuda()
                labels = labels.cuda()

                outputs = self.model(inputs)
                loss = entropy_regularized_loss(outputs, labels, self.entropy_weight)
                accuracy = top1_accuracy(outputs, labels)

                self.meters['test_loss'].add(loss.item())
                self.meters['test_accuracy'].add(accuracy.item())

            self.epoch += 1
            self.epoch_save()
                        
            if epoch in self.lr_step:
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 10

    def update(self):        
        valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.batch_size,
                                      shuffle=False, num_workers=4, pin_memory=True)

        y_pred = np.empty((0, len(self.valid_dataset.indices)))
        y_true = np.empty((0))

        self.model.eval()
        for inputs, labels in valid_dataloader:
            inputs = inputs.cuda()            

            y_pred = np.append(y_pred, self.model(inputs).detach().cpu().numpy(), axis=0)
            y_true = np.append(y_true, labels.numpy(), axis=0)

        update_indices = []
        update_indices_map = {}
        update_labels = []

        for label in np.random.permutation(range(y_pred.shape[1])):
            sort_indices = np.argsort(y_pred[:, label])[::-1]            
            sort_indices = sort_indices[~np.in1d(sort_indices, update_indices)]
            update_indices += list(sort_indices[:self.n_update])
            update_indices_map[label] = sort_indices[:self.n_update]
            update_labels.append(label)

        update_xs, update_ys = self.valid_dataset.get(update_indices)
        
        if self.pseudo_labels:
            update_ys = []
            for label in update_labels:
                update_ys += [self.train_dataset.backward_map[label][0]] * self.n_update
            update_ys = np.array(update_ys)

        self.train_dataset.add(update_xs, update_ys)
        self.valid_dataset.drop(update_indices_map)

        self.round += 1
        self.round_save()

    def run(self):        
        self.model.cuda()

        for round in range(self.n_rounds):
            self.train()
            self.update()

        self.environment_save()
