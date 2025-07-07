#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Training script for Point Transformer Networks.
    Created on Mon Jul 7 19:31 2025
    @author: STRH
    From Howsam 3D Computer Vision Course.
"""

# import libraries
from path import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics

from model import PCT
from dataset import PointCloudData, default_transforms, train_transforms

class AverageMeter(object):
    """
    Computes and stores the average and current value of loss
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(model, train_loader, test_loader, epochs=20):
    for epoch in range(epochs):
        model.train()
        loss_total = AverageMeter()
        accuracy = torchmetrics.Accuracy().cuda()

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs.transpose(1,2))
            loss = Loss(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_total.update(loss)
            accuracy(outputs.softmax(dim=-1), labels)
            # decrease batches
            if i==10:
                break
        acc = 100*accuracy.compute()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                outputs = model(inputs.transpose(1,2))
                _,predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if i==10:
                    break
        test_acc = 100*correct / total
        print('test accuracy: %d %%' % test_acc)

# define path
path = Path('Datasets/ModelNet10') 

# load data
train_ds = PointCloudData(path, transform=train_transforms())
test_ds = PointCloudData(path, valid=True, folder='test', transform=default_transforms())

# create dataloaders should be moved to train code
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PCT().cuda()
model.to(device)

# define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
Loss = nn.CrossEntropyLoss()

# train network
train(model, train_loader, test_loader)