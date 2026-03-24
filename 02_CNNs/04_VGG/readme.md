## 🏛 Architecture Overview

The VGG network (proposed by the Visual Geometry Group at Oxford) was a milestone in deep learning. Its primary contribution was demonstrating that **depth** is a critical component for high-performance visual recognition, achieved by using very small (**3x3**) convolution filters consistently throughout the network.

### 🍱 The VGG16 "Block" Strategy
VGG16 consists of **13 convolutional layers** and **3 fully connected layers**. It is organized into 5 functional blocks:

| Block | Layers | Output Channels | Feature Map Size (Input 224x224) |
| :--- | :--- | :---: | :---: |
| **Block 1** | 2 x Conv(3x3) + MaxPool | 64 | 112 x 112 |
| **Block 2** | 2 x Conv(3x3) + MaxPool | 128 | 56 x 56 |
| **Block 3** | 3 x Conv(3x3) + MaxPool | 256 | 28 x 28 |
| **Block 4** | 3 x Conv(3x3) + MaxPool | 512 | 14 x 14 |
| **Block 5** | 3 x Conv(3x3) + MaxPool | 512 | 7 x 7 |
| **Classifier** | 3 x FC (4096, 4096, 101) | - | 1 x 1 |

> [!TIP]
> **Why 3x3 filters?** A stack of two 3x3 conv layers (without spatial pooling in between) has an effective receptive field of 5x5; three such layers have a 7x7 effective receptive field. This allows the network to be deeper with fewer parameters compared to using larger filters.

---

## 🚀 Key Implementation Details

In the [VGG16.ipynb](VGG16.ipynb) notebook, several modern training techniques were implemented to ensure convergence and performance:

### ⚙️ Hyperparameters
- **Input Size**: 224x224x3
- **Batch Size**: 128
- **Learning Rate**: 0.01 (SGD with Momentum 0.9)
- **Epochs**: 300 (with Early Stopping)
- **Dropout**: 0.5 (used in the classifier)
- **Weight Decay**: 5e-4

### 🛠 Modern Training Enhancements
- **AMP (Automatic Mixed Precision)**: Utilizing `torch.amp` for faster training and reduced memory footprint.
- **Learning Rate Scheduler**: `ReduceLROnPlateau` to dynamically adjust the LR when validation loss plateaus.
- **Early Stopping**: A custom class to prevent overfitting by monitoring validation loss (patience = 10).

---

## 🔍 Interpretability with Grad-CAM

To understand **what** the model is looking at when making a prediction, we implement **Grad-CAM** (Gradient-weighted Class Activation Mapping).

---

## 📊 Dataset: Food101
The model is trained on the **Food101** dataset, which contains 101,000 images of 101 different food categories. 
- **Train/Val Split**: 80/20 split from the original training set.
- **Data Augmentation**: `RandomResizedCrop`, `ColorJitter`, `RandomHorizontalFlip`, and `RandomErasing`.

---
