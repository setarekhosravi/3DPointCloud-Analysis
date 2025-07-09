#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Training script for PointNet.
    Created on Wed Jul 9 17:48 2025
    @author: STRH
    From 3D Data Science Book.
"""

# import libraries
import sys
from path import Path
from tqdm import tqdm

import torch
import torchmetrics
from torch.utils.data import DataLoader

from model import PointNet
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


def pointNetLoss(outputs, labels, m3x3, m64x64, alpha=0.0001):
    """
    Function uses several important parameters: 
    The networks predictions (outputs), the true labels (labels),
    the 3x3 transformation matrix from the input transform (m3x3), the 64x64
    transformation matrix from the feature transform (m64x64), and a
    regularization parameter alpha, which defaults to 0.0001
    """
    criterion = torch.nn.NLLLoss() # because the network uses logsoftmax
    bs = outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs, 1, 1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs, 1, 1)
    if outputs.is_cuda():
        id3x3 = id3x3.cuda()
        id64x64 = id64x64.cuda()
    diff3x3 = id3x3 - torch.bmm(m3x3, m3x3.transpose(1,2))
    dif64x64 = id64x64 - torch.bmm(m64x64, m64x64.transpose(1,2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3) + torch.norm(dif64x64)) / float(bs)

def train(model, train_loader, val_loader=None, epochs=20, save=True):
    best_acc = 0.0
    for epoch in range(epochs):
        pointnet.train()
        running_loss = 0.0
        loss_total = AverageMeter()
        accuracy = torchmetrics.Accuracy().cuda()
        
        train_bar = tqdm(train_loader, desc='Training', leave=False, dynamic_ncols=True, file=sys.stdout)
        for i, data in enumerate(train_bar, 0):
            inputs, labels = data['pointcloud'].to(device).float(),
            data['category'].to(device)
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = pointnet(inputs.transpose(1,2))
            loss = pointNetLoss(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()
            loss_total.update(loss)
            accuracy(outputs.softmax(dim=-1), labels)
            
            train_bar.set_postfix({
                'Loss': f'{loss_total.avg:.4f}',
                'Train Acc': f'{100*accuracy.compute():.2f}%'
            })

        acc = 100*accuracy.compute()
        print(f'Train Acc: {acc:.2f}%, Avg Loss: {loss_total.avg:.4f}')

        pointnet.eval()
        correct = total = 0
        # validation
        if val_loader:
            test_bar = tqdm(test_loader, desc='Testing', leave=False, dynamic_ncols=True, file=sys.stdout)
            with torch.no_grad():
                for i, data in enumerate(test_bar):
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    outputs, __, __ = pointnet(inputs.transpose(1,2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    test_bar.set_postfix({
                        'Test Acc': f'{100 * correct / total:.2f}%'
                    })
            val_acc = 100. * correct / total
            print('Valid accuracy: %d %%' % val_acc)
        
        # Save last model (full model)
        torch.save(model, save_path + 'last.pt')

        # Save best model (full model)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model, save_path + 'best.pt')
            print(f'✅ New best model saved with accuracy: {best_acc:.2f}%')


# define path
path = Path('Datasets/ModelNet10') 
save_path = "PointNet/result/"

# load data
train_ds = PointCloudData(path, transform=train_transforms())
test_ds = PointCloudData(path, valid=True, folder='test', transform=default_transforms())

# create dataloaders should be moved to train code
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pointnet = PointNet().cuda()
pointnet.to(device)

# define optimizer and loss function
optimizer = torch.optim.Adam(pointnet.parameters(), lr=0.001)

# train network
train(pointnet, train_loader, test_loader)