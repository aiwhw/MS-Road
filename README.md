# MS-Road: A Large-Scale Challenging Benchmark for Multispectral Road Object Detection

<div align="center">

**[中文](#概述) | [English](#overview)**

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset](https://img.shields.io/badge/Dataset-MS--Road-blue)](https://github.com/aiwhw/MS-Road)
[![Paper](https://img.shields.io/badge/Paper-ACM%20MM%202026-red)](https://github.com/aiwhw/MS-Road)

</div>

<p align="center">
  <img src="samples/scene_samples/adverse_weather.gif" width="32%">
  <img src="samples/scene_samples/low_illumination.gif" width="32%">
  <img src="samples/scene_samples/overexposure.gif" width="32%">
</p>

<p align="center">
  <em>MS-Road dataset samples: Adverse Weather, Low Light, Overexposure</em>
</p>

---

## Overview

**MS-Road** is the first large-scale multispectral dataset specifically designed for road object detection. It comprises **49,253** high-resolution images with **512,194** carefully annotated instances across **9** critical categories, covering **7** representative road types and **7** challenging scenarios.

### Key Features

- 🎯 **Large Scale**: 49,253 images, 512,194 instances
- 🌈 **8 Spectral Bands**: 395-950 nm wavelength range
- 📐 **High Resolution**: 1200 × 900 pixels
- 🚗 **9 Categories**: person, bike, car, bus, truck, tricycle, van, traffic light, traffic sign
- 🛣️ **7 Road Types**: expressways, highways, side roads, town roads, rural roads, village roads, mountain roads
- ⚡ **7 Challenging Conditions**: adverse weather, overexposure, low illumination, severe occlusion, extremely small objects, shadow interference, high-density traffic

---

## 概述

**MS-Road** 是首个专为道路目标检测设计的大规模多光谱数据集。它包含 **49,253** 张高分辨率图像，**512,194** 个精心标注的实例，涵盖 **9** 个关键类别，**7** 种代表性道路类型和 **7** 种挑战性场景。

### 主要特点

- 🎯 **大规模**：49,253张图像，512,194个实例
- 🌈 **8个光谱波段**：395-950 nm波长范围
- 📐 **高分辨率**：1200 × 900像素
- 🚗 **9个类别**：行人、自行车、轿车、公交车、卡车、三轮车、面包车、交通灯、交通标志
- 🛣️ **7种道路类型**：高速公路、快速路、支路、城镇道路、乡村道路、村庄道路、山路
- ⚡ **7种挑战性条件**：恶劣天气、过曝、低光照、严重遮挡、极小小目标、阴影干扰、高密度交通

---

## 📊 Dataset Statistics / 数据集统计

| Attribute | Value / 数值 |
|-----------|-------------|
| Total Images / 总图像数 | 49,253 |
| Training Images / 训练集 | 34,520 (70%) |
| Test Images / 测试集 | 14,733 (30%) |
| Total Instances / 总实例数 | 512,194 |
| Categories / 类别数 | 9 |
| Spectral Bands / 光谱波段 | 8 (395-950 nm) |
| Resolution / 分辨率 | 1200 × 900 |

---

## 📥 Download / 下载

### Baidu Netdisk (百度网盘)
- **Link**: https://pan.baidu.com/s/1INry2A1PY8H-GUL3pH9MAA
- **Extraction Code (提取码)**: `ywfs`

### Google Drive (Coming Soon)

For detailed download information, please see [download.txt](download.txt).

---

## 📁 Data Format / 数据格式

### File Format / 文件格式
- **Format**: NumPy array (.npy)
- **Shape**: `(8, 1200, 900)` - **(Channels, Width, Height)**
- ⚠️ **Important**: The shape is **(C, W, H) NOT (C, H, W)**

### Spectral Bands / 光谱波段

| Band | Wavelength | Name |
|------|-----------|------|
| 1 | 395 nm | Near-UV |
| 2 | 450 nm | Blue |
| 3 | 525 nm | Green |
| 4 | 590 nm | Yellow |
| 5 | 650 nm | Red |
| 6 | 730 nm | Red Edge |
| 7 | 850 nm | NIR-1 |
| 8 | 950 nm | NIR-2 |

---

## 📖 Supplementary Material / 补充材料

For detailed technical information, please refer to the **[Supplementary Material](Supplementary_Material/Supplementary%20Material%20for%20MS-Road.pdf)**.

### Content Overview / 内容概览

The supplementary material (6 pages) provides comprehensive technical details including:

- **📸 MAIA Camera Specifications**: Detailed spectral band configuration (395-950 nm) and spectral response curves
- **🗂️ Dataset Construction**: Motivation for multispectral imaging, acquisition setup, and preprocessing pipeline
- **✏️ Annotation Protocol**: Three-stage annotation workflow (pre-labeling → manual refinement → expert verification), category definitions, and inter-annotator agreement statistics
- **📊 Comprehensive Statistics**: Instance distributions across 7 road types and 9 categories
- **📈 Detailed Experiments**: Complete baseline results, model parameters/FLOPs comparison, and feature activation visualizations (RGB vs MSI)
- **🚀 Future Plans**: Dataset expansion roadmap (tunnels, bridges, more nighttime data) and community support initiatives

---

## 🔬 Benchmark Results / 基准测试结果

We evaluate **14** representative detectors on MS-Road, including:
- **Two-stage**: Faster R-CNN, Dynamic R-CNN, Cascade R-CNN
- **One-stage**: ATSS, CenterNet, FCOS, RetinaNet, TOOD, YOLOv11n, YOLOv13n
- **DETR-based**: Deformable DETR, DINO, DDQ, RT-DETR

### Top Performers / 最佳性能

| Model | mAP50 | mAP | mAP_s | mAP_m | mAP_l |
|-------|-------|-----|-------|-------|-------|
| **DDQ** | **77.8%** | **49.9%** | **32.9%** | **56.0%** | **63.4%** |
| DINO | 77.5% | 48.9% | 31.8% | 54.9% | 61.5% |
| RT-DETR | 76.5% | 49.1% | 32.4% | 54.5% | 57.9% |
| TOOD | 75.2% | 47.6% | 29.2% | 54.8% | 60.4% |

### MSI vs RGB Comparison / 多光谱 vs RGB对比

| Model | Input | mAP50 | mAP | mAP_s |
|-------|-------|-------|-----|-------|
| ATSS | RGB | 70.4% | 43.9% | 23.8% |
| ATSS | **MSI** | **71.5%** | **44.8%** | **25.3%** |
| DDQ | RGB | 77.2% | 48.7% | 31.0% |
| DDQ | **MSI** | **77.8%** | **49.9%** | **32.9%** |

**MSI consistently outperforms RGB**, especially for small objects and challenging conditions.

---

## 🎨 Dataset Samples / 数据集样本

<p align="center">
  <img src="samples/scene_samples/adverse_weather.gif" width="24%">
  <img src="samples/scene_samples/low_illumination.gif" width="24%">
  <img src="samples/scene_samples/overexposure.gif" width="24%">
  <img src="samples/scene_samples/shadow_interference.gif" width="24%">
</p>

<p align="center">
  <em>Detection results under challenging conditions</em>
</p>

<p align="center">
  <img src="samples/scene_samples/severe_occlusion.gif" width="32%">
  <img src="samples/scene_samples/severe_occlusion_02.gif" width="32%">
  <img src="samples/scene_samples/severe_occlusion_03.gif" width="32%">
</p>

<p align="center">
  <em>Occlusion scenarios on different road types</em>
</p>

---

## 📄 License / 许可证

This dataset is released under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](LICENSE).

本数据集根据 [知识共享署名4.0国际许可协议 (CC BY 4.0)](LICENSE) 发布。

---

## 📧 Contact / 联系方式

For questions or suggestions, please:
- Open an issue on [GitHub Issues](https://github.com/aiwhw/MS-Road/issues)
- Contact the authors

如有问题或建议，请通过以下方式联系：
- 在 [GitHub Issues](https://github.com/aiwhw/MS-Road/issues) 提交问题
- 联系论文作者

---

## 🙏 Acknowledgments / 致谢

We would like to thank all annotators and reviewers who contributed to this dataset.

我们要感谢所有为这个数据集做出贡献的标注人员和审稿人。

---

<div align="center">

**Beijing Institute of Technology** | 北京理工大学

</div>
