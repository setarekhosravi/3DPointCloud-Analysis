#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    This code is for dataset analysis.
    Espeially we use ModelNet10.
    For visualizing meshes use the Open3D_Basics.ipynb notebook in current repo.
    Created on Sat Jun 28 22:04 2025
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

# body of the code started 
path = Path('Datasets/ModelNet10')   
# print(path)

# we can use open3d to sample from the mesh to extract a point cloud
class PointSampler(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * (side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0) ** 0.5
    
    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t-s)*pt2[i] + (1-t)*pt3[i]
        return (f(0), f(1), f(2))
    
    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(faces)):
            areas[i] = (self.triangle_area(verts[faces[i][0]],
                                          verts[faces[i][1]],
                                          verts[faces[i][2]]))
            
        sampled_faces = random.choices(faces, weights=areas, cum_weights=None, k=self.output_size)
        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = self.sample_point(verts[sampled_faces[i][0]],
                                                  verts[sampled_faces[i][1]],
                                                  verts[sampled_faces[i][2]])
            
        return sampled_points

# normalize the point cloud
class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return norm_pointcloud

# augment the point cloud
class RandRotation_z(object):
    def __call__(self, pointcloud):
        assert  len(pointcloud.shape) == 2

        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[math.cos(theta), -math.sin(theta), 0],
                               [math.sin(theta), math.cos(theta), 0],
                               [0, 0, 1]])
        rotated_pointcloud = rot_matrix.dot(pointcloud.T).T
        return rotated_pointcloud
    
class RandomNoise(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        noise = np.random.normal(0, 0.02, (pointcloud.shape))
        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud

# convert to tensor
class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2

        return torch.from_numpy(pointcloud)
    
# define the dataset class using 3D Data Science book with minor changes (debugging some lines)
class PointCloudData(Dataset):
    def __init__(self, root_dir, valid=False, folder='train', transform=None):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform if not valid else default_transforms()
        self.valid = valid
        self.files = []
        for category in self.classes.keys():
            new_dir = root_dir/Path(category)/folder
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = {}
                    sample['pcd_path'] = new_dir/file
                    sample['category'] = category
                    self.files.append(sample)
    
    def __len__(self):
        return len(self.files)
    
    def __preproc__(self, file):
        verts, faces = read_off(file)
        if self.transforms:
            pointcloud = self.transforms((verts, faces))
        return pointcloud
    
    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        category = self.files[idx]['category']
        with open(pcd_path, 'r') as f:
            pointcloud = self.__preproc__(f)
        return {'pointcloud': pointcloud,
            'category': self.classes[category]}
        
# list all objects in the dataset
def default_transforms():
    return transforms.Compose([
                                PointSampler(1024),
                                Normalize(),
                                ToTensor()
                            ])

def train_transforms():
    return transforms.Compose([
                                PointSampler(1024),
                                Normalize(),
                                RandRotation_z(),
                                RandomNoise(),
                                ToTensor()
                            ])
    
# read .off files contains vertices and faces
def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, _ = tuple([int(item) for item in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces

# visualize the mesh
def visualize_rotate(data):
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames=[]

    def rotate_y(x, y, z, theta):
        w = x+1j*y
        return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z
    
    for t in np.arange(0, 10.26, 0.1):
        xe, ye, ze = rotate_y(x_eye, y_eye, z_eye, -t)
        frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))
    
    fig = go.Figure(data=data,
                    layout=go.Layout(
                        updatemenus=[dict(type='buttons',
                                          showactive=False,
                                          y=1,
                                          x=0.8,
                                          xanchor='left',
                                          yanchor='bottom',
                                          pad=dict(t=45, r=10),
                                          buttons=[dict(label='Play',
                                                        method='animate',
                                                        args=[None, dict(frame=dict(duration=50, redraw=True),
                                                                         transition=dict(duration=0),
                                                                         fromcurrent=True,
                                                                         mode='immediate')]
                                                        )]
                                            )]
                                    ), 
                                    frames=frames
                    )
    return fig

# visualize the point cloud
def pcshow(xs,ys,zs):
    data = go.Figure(data=[go.Scatter3d(x=xs, y=ys, z=zs,
                                        mode='markers')])
    fig = visualize_rotate(data)
    fig.update_traces(marker=dict(size=2,
                                  line=dict(color='darkblue', width=2)), 
                                  selector=dict(mode='markers'))
    fig.show()

# body of code to test the code
if __name__ == "__main__":
    with open(path/'chair/train/chair_0001.off', 'r') as f:
        verts, faces = read_off(f)

    i,j,k = np.array(faces).T
    x,y,z = np.array(verts).T

    visualize_rotate([go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, opacity=0.5)]).show()
    pcshow(x,y,z)

    pointcloud = PointSampler(2048)([verts, faces])
    pcshow(*pointcloud.T)

    norm_pointcloud = Normalize()(pointcloud)
    pcshow(*norm_pointcloud.T)

    rot_pointcloud = RandRotation_z()(norm_pointcloud)
    noisy_rot_pointcloud = RandomNoise()(rot_pointcloud)
    pcshow(*noisy_rot_pointcloud.T)

    train_ds = PointCloudData(path, transform=train_transforms())
    test_ds = PointCloudData(path, valid=True, folder='test', transform=default_transforms())

    print('Train dataset size:', len(train_ds))
    print('Test dataset size:', len(test_ds))
    print('Number of classes:', len(train_ds.classes))
    print("Sample pointcloud shape:", train_ds[0]['pointcloud'].shape)

    data = train_ds[2000]
    pcshow(*data['pointcloud'].T)
    
    # create dataloaders should be moved to train code
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)