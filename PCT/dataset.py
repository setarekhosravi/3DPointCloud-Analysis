#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    This code is for dataset analysis.
    Espeially we use ModelNet10.
    For visualizing meshes use the Open3D_Basics.ipynb notebook in current repo.
    Created on Sat Jun 28 22:04 2025
    @author: STRH
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
import torch.nn.functional as f
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils
import torchmetrics

import plotly.graph_objects as go
import plotly.express as px

# body of the code started 
path = Path("Datasets/ModelNet10")   
# print(path)

# read .off files contains vertices and faces
def read_off(file):
    n_verts, n_faces, _ = tuple([int(item) for item in file.readline().strip().split(' ')])
    verts = []