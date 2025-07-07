#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    This code contains Point Transformer Networks.
    Created on Mon Jul 7 15:32 2025
    @author: STRH
    From Howsam 3D Computer Vision Course.
"""

# import libraries
import numpy as np
import math
import random
import os
import torch
import scipy.spatial.distance
from path import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils
import torchmetrics

import plotly.graph_objects as go
import plotly.express as px

import pointnet2_ops
from pointnet2_ops.pointnet2_utils import pointnet2_utils

# Point cloud transformer class
class PCT(nn.Module):
    def __init__(self):
        super(PCT, self).__init__()

        # input embedding (Green in Fig 4 of paper)
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(64)

        # input embedding (blue in Fig 4 of paper which has LBR)
        self.mini1 = MiniPointNet()
        self.mini2 = MiniPointNet()

    def forward(self, x):
        # input embedding (Green in Fig 4 of paper)
        x = F.relu(self.bn1(self.conv1(x))) # LBR 1
        x = F.relu(self.bn2(self.conv2(x))) # LBR 2
        # input embedding (blue in Fig 4 of paper which sampling and grouping)
        # sampling and grouping does not have any learnable parameter so should
        # be implemented in forward function
        # use pointnet2_ops_lib: clone and install through git://github.com/erikwijmans/Pointnet2_Pytorch.git
        return x