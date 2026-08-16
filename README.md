# CNN Image Classification — Geometric Shapes (Circle / Square / Triangle)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahsanjust/CNN-Geometric-Shapes-Classification/blob/master/220119_CNN.ipynb)

**Student Name:** Ahsanul Haque  
**Student ID:** 220119  
**Course:** Artificial Intelligence & Machine Learning Lab (CSE 3-2) — Assignment 8 (Final Lab)

---

## 1. Overview
This project implements a comprehensive **Convolutional Neural Network (CNN)** pipeline in **PyTorch** for classifying 2D **Geometric Shapes** (Dataset Option 5: **Circles**, **Squares**, and **Triangles**). 

The model is trained on a standard split (`training/train`, `training/validation`, `training/test`) and evaluated on **10 real-world smartphone camera photographs** of hand-drawn shapes on paper (`dataset/`). 

### Zero-Manual-Intervention Colab Automation:
Opening the notebook in Google Colab and clicking **Runtime → Run all** automatically:
1. Clones the repository to retrieve dataset splits and custom photos.
2. Loads the saved model weights (`model/220119.pth`) if present (or trains in ~10 seconds on CPU).
3. Evaluates the test set, builds the confusion matrix, and visualizes misclassified samples.
4. Performs inference on the 10 real-world phone photos, outputting prediction confidence scores.
5. Generates all 5 evaluation plots with zero manual file uploads.

---

## 2. Dataset Architecture

### Standard Training Split (`training/`)
A structured drawn-shape dataset organized using `torchvision.datasets.ImageFolder`:
- **Train Set (543 images):** 304 Circles, 103 Squares, 136 Triangles
- **Validation Set (180 images):** 60 Circles, 60 Squares, 60 Triangles
- **Test Set (90 images):** 30 Circles, 30 Squares, 30 Triangles

```text
training/
├── train/       (543 images: 304 circles / 103 squares / 136 triangles)
├── validation/  (180 images: 60 circles / 60 squares / 60 triangles)
└── test/        ( 90 images: 30 circles / 30 squares / 30 triangles)
```

Class names are dynamically mapped from the folder structure (`0: circles`, `1: squares`, `2: triangles`) to prevent hardcoding integer labels.

### Custom Real-World Smartphone Dataset (`dataset/`)
- 10 smartphone photos (4 Circles, 3 Squares, 3 Triangles) drawn on paper with a ballpoint pen.
- Preprocessed to remove camera watermark overlays and binary metadata headers.
- Centered into 1:1 square aspect ratio bounding boxes to preserve geometric symmetry during downsampling.

---

## 3. Model Architecture & Preprocessing

### Single Shared Transform Pipeline
Both training images and real-world test images pass through the identical transform pipeline:
$$\text{Resize}((64, 64)) \to \text{ToTensor}() \to \text{Normalize}(\mu=[0.5, 0.5, 0.5], \sigma=[0.5, 0.5, 0.5])$$

### CNN Architecture (`nn.Module`):
- **Conv Layer 1:** `nn.Conv2d(3, 32, kernel_size=3, padding=1)` $\to$ `nn.ReLU()` $\to$ `nn.MaxPool2d(2, 2)` $\to$ Output: $(32, 32, 32)$
- **Conv Layer 2:** `nn.Conv2d(32, 64, kernel_size=3, padding=1)` $\to$ `nn.ReLU()` $\to$ `nn.MaxPool2d(2, 2)` $\to$ Output: $(64, 16, 16)$
- **Flatten Layer:** Reshape $(64 \times 16 \times 16) = 16,384$ features.
- **Dense Layer 1:** `nn.Linear(16384, 128)` $\to$ `nn.ReLU()`
- **Output Layer:** `nn.Linear(128, 3)` (logits for 3 classes)
- **Total Learnable Parameters:** $2,098,059$

---

## 4. Experimental Results & Visualizations

### 1) Training History
- **Optimizer:** `torch.optim.Adam(lr=0.001)` | **Loss:** `nn.CrossEntropyLoss` | **Batch Size:** 64 | **Epochs:** 10
- **Final Train Loss:** 0.1236 | **Train Accuracy:** 97.42%
- **Final Validation Loss:** 0.1569 | **Validation Accuracy:** 96.67%
- **Test Set Accuracy:** **95.56% (86 / 90 correct)**

| Accuracy vs. Epochs | Loss vs. Epochs |
| :---: | :---: |
| ![Accuracy vs Epochs](results/AvE.png) | ![Loss vs Epochs](results/LvE.png) |

---

### 2) Confusion Matrix & Error Diagnostics
- **Circles:** 27 / 30 correct (90.0% recall)
- **Squares:** 29 / 30 correct (96.7% recall)
- **Triangles:** 30 / 30 correct (100.0% recall)

| Test Confusion Matrix | Visual Error Analysis |
| :---: | :---: |
| ![Confusion Matrix](results/confusion_matrix.png) | ![Error Analysis](results/error_analysis.png) |

---

### 3) Real-World Smartphone Photo Predictions
The trained CNN model was evaluated on the 10 custom smartphone photographs:

![Custom Phone Predictions](results/custom_predictions.png)

#### Detailed Phone Photo Prediction Breakdown:
| Filename | True Shape | Predicted Class | Confidence (%) | Class Probabilities $[C, S, T]$ | Result |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `circle1.jpg` | **Circle** | **circles** | 97.4% | $[97.4\%, 0.9\%, 1.6\%]$ | ✅ Correct |
| `circle2.jpg` | **Circle** | **circles** | 97.1% | $[97.1\%, 1.4\%, 1.5\%]$ | ✅ Correct |
| `circle3.jpg` | **Circle** | **circles** | 94.4% | $[94.4\%, 4.1\%, 1.5\%]$ | ✅ Correct |
| `circle4.jpg` | **Circle** | **circles** | 92.0% | $[92.0\%, 7.5\%, 0.5\%]$ | ✅ Correct |
| `square1.jpg` | **Square** | **squares** | 65.0% | $[3.0\%, 65.0\%, 32.0\%]$ | ✅ Correct |
| `square2.jpg` | **Square** | **circles** | 61.6% | $[61.6\%, 31.1\%, 7.2\%]$ | ❌ Curved Stroke Ambiguity |
| `square3.jpg` | **Square** | **squares** | 79.9% | $[15.6\%, 79.9\%, 4.6\%]$ | ✅ Correct |
| `triangle1.jpg`| **Triangle** | **triangles** | 97.7% | $[2.1\%, 0.2\%, 97.7\%]$ | ✅ Correct |
| `triangle2.jpg`| **Triangle** | **triangles** | 98.7% | $[1.0\%, 0.3\%, 98.7\%]$ | ✅ Correct |
| `triangle3.jpg`| **Triangle** | **triangles** | 94.4% | $[3.9\%, 1.7\%, 94.4\%]$ | ✅ Correct |

**Overall Real-World Accuracy:** **9 / 10 (90.0%)**

---

## 5. Scientific Domain Shift Analysis

1. **Stroke Width & Pixel Density Disparity:**
   The standard training dataset comprises clipart-style drawings with thick lines ($10\text{--}20\text{ px}$ wide) on solid white backgrounds. In contrast, custom smartphone photos feature thin ballpoint pen lines ($1\text{--}2\text{ px}$ wide). At $64 \times 64$ resolution, thin lines exhibit subtle contrast loss against natural paper shadows.

2. **Geometric Centering Impact:**
   Applying 1:1 contour-based square centering ensured that the angular vertices of triangles and right angles of squares were preserved symmetrically within the convolutional receptive fields, yielding $>94\%$ confidence across all triangles and circles.

3. **Authentic Edge Ambiguity in `square2.jpg`:**
   In `square2.jpg`, hand-drawn outward curvature on the vertical strokes softened corner transitions. When downsampled to $64\times 64$, the continuous boundary gradient resembled circular curvature, causing the model to predict Circle ($61.6\%$) over Square ($31.1\%$), providing a verifiable edge-case failure mode.

---

## 6. How to Run Locally

```bash
# Clone the standalone repository
git clone https://github.com/ahsanjust/CNN-Geometric-Shapes-Classification.git
cd CNN-Geometric-Shapes-Classification

# Run the notebook
jupyter notebook 220119_CNN.ipynb
```
