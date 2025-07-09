#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Training script for PointNet.
    Created on Wed Jul 9 17:48 2025
    @author: STRH
    From 3D Data Science Book.
"""

# import libraries
from path import Path

import torch
from torch.utils.data import DataLoader

from model import PointNet
from dataset import PointCloudData, default_transforms, train_transforms

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
    for epoch in range(epochs):
        pointnet.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(),
            data['category'].to(device)
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = pointnet(inputs.transpose(1,2))
            loss = pointNetLoss(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            
            if i % 10 == 9:
                # print every 10 mini-batches
                print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                     (epoch + 1, i + 1, len(train_loader),
                      running_loss / 10))
                running_loss = 0.0

    pointnet.eval()
    correct = total = 0
    # validation
    if val_loader:
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                outputs, __, __ = pointnet(inputs.transpose(1,2))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = 100. * correct / total
        print('Valid accuracy: %d %%' % val_acc)
    
    # save the model
    if save:
        torch.save(pointnet.state_dict(), "save_"+str(epoch)+".pth")

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