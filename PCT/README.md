# 🧠 Point Cloud Transformer (PCT)

This directory contains a PyTorch implementation of the **Point Cloud Transformer (PCT)**, designed for classification of 3D point cloud data. The model is trained and evaluated on the [ModelNet10](http://modelnet.cs.princeton.edu/) dataset.

## 📌 Overview

The Point Cloud Transformer architecture was introduced in the paper:

> **"Point Cloud Transformer"**  
> Hengshuang Zhao, Li Jiang, Jiaya Jia, Philip Torr, Vladlen Koltun  
> [CVPR 2021] [[Paper](https://arxiv.org/abs/2012.09688)]

![Architecture of PCT](/Images/pct.png)

Unlike conventional methods such as PointNet/PointNet++, PCT leverages a transformer-based self-attention mechanism to better model spatial relationships between 3D points.

### 🔧 Core Components
- **MiniPointNet** for local feature extraction
- **Farthest point sampling and grouping** to capture neighborhoods
- **Stacked attention blocks** (4 layers) for global context modeling
- **Final classifier** for shape category prediction

---

## 🧪 Training

You can train the model using the following command:

```bash
python train.py
````

> Make sure your dataset folder (e.g., `ModelNet10`) is placed inside the `Datasets/` directory.

During training:

* Best model is saved as `result/best.pt`
* Last model is saved as `result/last.pt`
* Training uses 20 epochs by default with Adam optimizer

### ✅ Best Validation Accuracy

| Epochs | Best Accuracy |
| ------ | ------------- |
| 20     | **51.65%**    |

---

## 🔍 Inference

To run inference on a single `.off` file and visualize it using Open3D with the predicted class:

```bash
python inference.py --input path/to/sample.off --weights result/best.pt
```

* Output: Class name printed in terminal
* Visualization: Colored point cloud shown in Open3D window

---

## 🧠 Note

* Ensure that all utility functions and modules are available (like `sample_and_group`, `PointSampler`, and `read_off`).
* Visualization in inference relies on Open3D (`v0.19.0`), make sure it's installed.

---

## 🧾 Acknowledgment

This implementation is part of a learning project and inspired by:

* 📘 Howsam Academy Course on Computer Vision — *Seyed Sajad Ashrafi*
* 📘 3D Data Science — *Florent Poux*

```