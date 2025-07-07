#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Inference script for Point Cloud Classification using PCT and Open3D
    Created on Sat Jun 28 22:04 2025
    @author: STRH
    From Howsam 3D Computer Vision Course.
"""

import torch
import numpy as np
import argparse
import open3d as o3d
from model import PCT
from dataset import PointSampler, read_off, default_transforms

# Class label map (update this based on your dataset)
class_map = {
    0: 'bathtub',
    1: 'bed',
    2: 'chair',
    3: 'desk',
    4: 'dresser',
    5: 'monitor',
    6: 'night_stand',
    7: 'sofa',
    8: 'table',
    9: 'toilet'
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_open3d(pointcloud_np, title=""):
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud_np)
    pcd.paint_uniform_color([1, 0, 0])  # red

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Prediction: {title}", width=800, height=600)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

def infer(file_path, model_path):
    # Load and preprocess point cloud
    with open(file_path, 'r') as f:
        verts, faces = read_off(f)
    pointcloud = PointSampler(2048)([verts, faces])
    transform = default_transforms()
    sample = {'pointcloud': pointcloud, 'category': 0}
    sample = transform(sample)
    pc_tensor = sample['pointcloud'].unsqueeze(0).to(device).float()  # [1, N, 3]

    # Load model (entire model object)
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(pc_tensor.transpose(1, 2))  # [1, num_classes]
        pred_class = torch.argmax(output, dim=1).item()
        class_name = class_map[pred_class]

    print(f"✅ Predicted class: {class_name}")
    visualize_open3d(pc_tensor.squeeze(0).cpu().numpy(), title=class_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Point Cloud Inference with Open3D")
    parser.add_argument('--input', '-i', required=True, help='Path to the .off file')
    parser.add_argument('--weights', '-w', default='best.pt', help='Path to the model weights (full model)')

    args = parser.parse_args()

    infer(args.input, args.weights)
