# Conditional-GAN-Sketch-to-Image-Translation

## 📌 Project Overview

This project implements a Pix2Pix-based Image-to-Image Translation GAN designed to transform sketch/edge-like inputs into realistic colored images.
It includes end-to-end functionality: synthetic dataset generation, GAN training, rich visualization, and inference on single or batch images.

The system is built entirely in PyTorch, uses a U-Net Generator and a PatchGAN Discriminator with Spectral Normalization, and provides deep interpretability through latent space analysis and discriminator heatmaps.

## 🚀 Key Features

### 1️⃣ Synthetic Dataset Generation
- Fully automated paired dataset creation
- Input: Sketches, edges, geometric shapes, window grids
- Output: Colored and textured images
- Train/test split handled automatically
- No external datasets required

### 2️⃣ Pix2Pix GAN Architecture

#### 🔹 Generator (U-Net)
- Encoder–decoder architecture with skip connections
- 8 down-sampling + 7 up-sampling layers
- Instance Normalization for stable training
- Dropout for regularization
- Tanh activation for output normalization

#### 🔹 Discriminator (PatchGAN)
- Operates on local image patches instead of full images
- Spectral Normalization for training stability
- Instance Normalization + LeakyReLU activations
- Outputs realism scores for each image patch

### 3️⃣ Robust Training Pipeline
- Adversarial loss (LSGAN / MSE-based)
- L1 reconstruction loss with high weighting
- Separate learning rates for Generator and Discriminator
- Periodic checkpoint saving
- Automatic sample generation during training
- Designed to run on **CPU or GPU**

### 4️⃣ Advanced Visualization & Explainability
- Comprehensive training dashboard including:
  - Input vs Target vs Generated images
  - Pixel-wise error heatmaps
  - Side-by-side qualitative comparisons
  - RGB channel distribution analysis
- Discriminator analysis:
  - Patch-wise confidence heatmaps
  - Real vs fake score distributions
- Latent space visualization:
  - Bottleneck feature inspection
  - Channel-wise activation strength
  - Identification of dominant latent features
- Automatic training quality scoring  
  *(Excellent / Good / Fair / Poor)*

### 5️⃣ Flexible Inference Modes
- **Single image inference**
- **Batch image inference**
- Supports:
  - Input-only images
  - Combined input | target images
- Saves:
  - Generated images
  - Full diagnostic visualization reports
  - Discriminator confidence metrics

## 🛠️ Tech Stack

### Core Technologies
- **Python**
- **PyTorch**
- **TorchVision**

### Deep Learning Models
- Pix2Pix GAN
- U-Net Generator
- PatchGAN Discriminator

### Training & Optimization
- Adam Optimizer
- L1 + Adversarial (LSGAN) Loss
- Spectral Normalization
- Instance Normalization

### Data & Image Processing
- NumPy
- PIL (Pillow)
- torchvision.transforms

### Visualization & Analysis
- Matplotlib
- Custom GAN dashboards
- Latent space and feature activation analysis

### Utilities
- tqdm (progress tracking)
- OS & file system utilities

---

## 📂 Project Structure

```text
.
├── datasets/
│   └── synthetic/
│       ├── train/                 # Synthetic training images
│       └── test/                  # Synthetic test images
│
├── outputs/
│   ├── checkpoints/               # Model checkpoints
│   ├── samples/                   # Training sample outputs
│   ├── visualizations/            # Training visual dashboards
│   └── custom_results/            # Inference results
│
├── app.py                         # Main training & inference script
└── README.md
