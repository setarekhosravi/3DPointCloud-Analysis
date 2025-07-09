# 🔷 PointNet — 3D Point Cloud Classification

This directory contains a PyTorch implementation of the **PointNet** architecture for 3D object classification on the ModelNet10 dataset. This work is part of my 3D Vision learning journey.

---

## 📘 Paper Overview

> **PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation**  
> Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas  
> [CVPR 2017] [[Paper](https://arxiv.org/abs/1612.00593)]

![Architecture of PointNet](../Images/pointnet.png)

**PointNet** is a landmark architecture designed to directly consume raw, unordered 3D point clouds. It ensures permutation invariance using:
- Shared MLP layers for feature extraction,
- A symmetric function (max pooling) for global aggregation,
- And **T-Net modules** to learn affine transformations on both input and feature spaces for geometric invariance.

> ✅ This implementation includes both the **input T-Net (3×3)** and **feature T-Net (64×64)**.

---

## 🚀 Training

To train the model:

```bash
python train.py
```

* Trains on ModelNet10 for 20 epochs by default
* Uses `NLLLoss` with `log_softmax` outputs
* Includes regularization terms from transformation matrices
* Saves:

  * `result/last.pt`: the latest model
  * `result/best.pt`: the model with highest validation accuracy

### ✅ Best Validation Accuracy

| Epochs | Best Accuracy |
| ------ | ------------- |
| 20     | **82.93**    |

> 📂 Make sure the dataset (e.g., `ModelNet10`) is located under the `Datasets/` directory.

---

## 🔍 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Inference

Run inference on a single `.off` file using:

```bash
python inference.py --input path/to/file.off --weights result/best.pt
```

Example:

```bash
python inference.py -i ../Datasets/ModelNet10/chair/test/chair_0011.off
```

The script:

* Predicts the class of the point cloud
* Displays the result in the terminal
* Visualizes the point cloud using Open3D

---

## 🛠️ Dependencies

Install all dependencies using the `requirements.txt` in the main repository.

Key packages:

* `torch`, `torchvision`
* `torchmetrics`
* `open3d`
* `tqdm`

---

## 📚 References

This implementation is based on:

* 📘 *3D Data Science* by Florent Poux
* 🎓 Howsam Academy Course on Computer Vision by Seyed Sajad Ashrafi
