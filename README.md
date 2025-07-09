# 🌀 PointCloud Analysis

This repository contains my learning journey and implementation of 3D vision algorithms, with a focus on **point cloud classification** using deep learning in PyTorch and visualization using Open3D.

---

## 📁 Repository Structure

```

PointCloud Analysis/
├── Datasets/
│   └── ModelNet10/           ← Place your dataset folders here
├── PCT/                      ← Point Cloud Transformer (PCT) implementation
├── PointNet/                 ← PointNet implementation
├── Open3D\_Basics.ipynb       ← Introductory tutorial on Open3D
├── Open3D\_SurfaceReconstruction.ipynb ← Surface reconstruction with Open3D
├── README.md                 

```

---

## 🔍 Overview

### 🔷 `PCT/` — Point Cloud Transformer 
An implementation of the **Point Cloud Transformer** network for point cloud classification. Includes:
- Full training and inference pipelines
- Transformer-based attention architecture
- Open3D visualization support
- See the [paper](https://arxiv.org/abs/2012.09688)

📄 See `PCT/README.md` for full details.

---

### 🔷 `PointNet/` — PointNet
An implementation of the classic **PointNet** network, adapted from the 3D Data Science book. Includes:
- T-Net based feature transformation
- NLL loss with orthogonality regularization
- Inference with Open3D rendering
- See the [paper](https://arxiv.org/abs/1612.00593)

📄 See `PointNet/README.md` for more info.

---

### 🔶 `Datasets/`
Expected to contain datasets such as [ModelNet10](http://modelnet.cs.princeton.edu/), organized in the format:
```

Datasets/
└── ModelNet10/
├── bed/
│   └── train/
│   └── test/
└── chair/
...

````

---

### 📘 Jupyter Notebooks

- `Open3D_Basics.ipynb`: A tutorial notebook to get started with point cloud and mesh reading, visualization using Open3D.
- `Open3D_SurfaceReconstruction.ipynb`: Surface reconstruction from raw point cloud data.

---

## 🚀 How to Start

```bash
# Clone the repo
git clone https://github.com/your-username/pointcloud-analysis.git
cd pointcloud-analysis

# Install dependencies
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# Put ModelNet10 or other datasets in the Datasets/ directory
````

## ✅ Future Work

* Expand Open3D tutorials
* Add PointNet++ or DGCNN
* Support segmentation tasks

---

## 🙏 Acknowledgments

This repository is the result of my learning and exploration in the field of 3D vision and point cloud analysis. It has been built upon the knowledge and inspiration provided by the following resources:

- 📚 **Hamrah (MCI) Academy Course on Computer Vision** by *Dr. Shohreh Kasaei*
- 🎓 **Howsam Academy Course on Computer Vision** by *Seyed Sajad Ashrafi*
- 📘 *3D Data Science* by *Dr. Florent Poux*
- 📗 *3D Point Cloud Analysis* by *Shan Liu* and co-authors

I am deeply grateful to these educators and authors for their excellent teaching and materials.

---

Feel free to explore each subfolder’s README for more detailed usage and results.