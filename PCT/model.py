#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    This code contains Point Transformer Networks.
    Created on Mon Jul 7 15:32 2025
    @author: STRH
    From Howsam 3D Computer Vision Course.
"""

# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import sample_and_group

# mini pointnet class
class MiniPointNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MiniPointNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self,x): # [B, N, S, C] = [32, 512, 32, 128]
        b, n, s, c = x.size()
        x = x.permute(0, 1, 3, 2) # [32, 512, 128, 32]
        x = x.reshape(-1, c, s) # [32*512, 128, 32]
        x = F.relu(self.bn1(self.conv1(x))) 
        x = F.relu(self.bn2(self.conv2(x))) 
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1) # [32*512, 128]
        x = x.reshape(b, n, c) # [32, 512, 128]
        # x = x.permute(0, 2, 1) # [32, 128, 512]
        return x
# attention class
class Attention(nn.Module):
    def __init__(self, channels):
        super(Attention, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, kernel_size=1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, kernel_size=1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.trans_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        energy = torch.bmm(x_q, x_k)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))

        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

# Transformer class
class Transformer(nn.Module):
    def __init__(self, channels=256):
        super(Transformer, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        # self attention layers
        self.sa1 = Attention(channels)
        self.sa2 = Attention(channels)
        self.sa3 = Attention(channels)
        self.sa4 = Attention(channels)

    def forward(self, x):
        # b, 3, npoints, nsample
        # conv2d 3-> 128 channels 1, 1
        # b * npoints, c, nsample
        # permute reshape
        batch_size, _, N = x.size()

        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        return x
    
# Point cloud transformer class
class PCT(nn.Module):
    def __init__(self):
        super(PCT, self).__init__()

        # input embedding (Green in Fig 4 of paper) LBR
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(64)

        # input embedding (blue in Fig 4 of paper which has LBR)
        self.mini1 = MiniPointNet(128,128)
        self.mini2 = MiniPointNet(256,256)
        
        # transformer
        self.transformer = Transformer(256)

        # LBR
        self.conv3 = nn.Conv1d(1024, 1024, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(1024)
        
        # Classifier
        self.cls = nn.Sequential(nn.Linear(1024, 512),
                                 nn.LayerNorm(512),
                                 nn.LeakyReLU(negative_slope=0.2),
                                 nn.Linear(512, 256),
                                 nn.LayerNorm(256),
                                 nn.LeakyReLU(negative_slope=0.2),
                                 nn.Linear(256, 10))

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        # input embedding (Green in Fig 4 of paper)
        x = F.relu(self.bn1(self.conv1(x))) # LBR 1
        x = F.relu(self.bn2(self.conv2(x))) # LBR 2
        # input embedding (blue in Fig 4 of paper which sampling and grouping)
        # sampling and grouping does not have any learnable parameter so should
        # be implemented in forward function
        # use pointnet2_ops_lib: clone and install through git://github.com/erikwijmans/Pointnet2_Pytorch.git
        x = x.permute(0, 2, 1)
        new_xyz, new_points = sample_and_group(npoint=512, radius=0.15, nsample=32, xyz=xyz, points=x)
        new_points = self.mini1(new_points)
        new_xyz, new_points = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=new_points)
        new_points = self.mini2(new_points) # [b, n, c]
        new_points = new_points.permute(0, 2, 1)
        feature = self.transformer(new_points)
        # To Do: Concat
        # LBR
        feature = F.relu(self.bn3(self.conv3(feature)))
        feature = F.adaptive_max_pool1d(feature, 1).squeeze(-1)
        # Classifier
        feature = self.cls(feature)
        return feature