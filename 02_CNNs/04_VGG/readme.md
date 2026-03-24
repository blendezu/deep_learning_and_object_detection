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

## 🏛️ VGG16 vs. AlexNet: The Evolution of Depth

While **AlexNet (2012)** proved that Deep Learning worked, **VGG16 (2014)** proved that **Systematic Depth** was the key to scaling performance.

| Feature | AlexNet | VGG16 |
| :--- | :--- | :--- |
| **Layers** | 8 layers | 16 layers |
| **Filter Sizes** | Large & Varied (11x11, 5x5, 3x3) | **Uniform 3x3** Throughout |
| **Architecture** | Heuristic (hand-tuned) | **Modular Blocks** (standardized units) |
| **Non-Linearity** | Fewer activations | More ReLUs (increased discriminative power) |

**The VGG Breakthrough:**
VGG replaced large filters (like 7x7) with a stack of three 3x3 filters. This achieved the same "receptive field" but introduced **more non-linearities** (ReLU) and **fewer parameters**, making the model's feature extraction more expressive.

---

## 💾 The OOM Problem: VGG16 vs. AlexNet

I had so much problem of Memory while training VGG16, but not on AlexNet. So I didnt train it. But the script ran. Here might be the reason why:

### 1. Activation Volume (The Real Culprit)
Memory consumption during training isn't just about parameters; it's mostly about **Activations** (intermediate feature maps) that must be stored for backpropagation.
*   **AlexNet** starts with a `Stride=4` in the first layer. This immediately reduces the input (227x227) to 55x55.
*   **VGG16** preserves the full resolution (224x224) for several layers before the first MaxPool. Storing 64 channels of 224x224 floats takes **significantly** more RAM than AlexNet's early-reduced maps.

### 2. Depth and Gradient Storage
VGG16 is twice as deep (16 layers vs 8). During the backward pass, PyTorch must keep the gradients and activations for all 16 layers in memory. 

### 3. Parameter Count
*   **AlexNet**: ~61 Million parameters.
*   **VGG16**: ~138 Million parameters.
*   With a larger model, the **Optimizer States** (which store momentum and gradients for every single parameter) also double in size.

---

## ⚠️ VGG16 Obsolete Now

Despite its historical importance, VGG16 is rarely used in production today for three main reasons:

1.  **Massive Parameter Weight**: Being said VGG16 has **138 million parameters**, primarily due to the massive Fully Connected (FC) layers. A modern **ResNet-50** has only ~25M parameters while achieving higher accuracy.
2.  **Vanishing Gradients**: VGG16 lacks "Skip Connections." As networks get deeper than 20 layers without these connections, they become extremely difficult to train.
3.  **Computational Inefficiency**: It is slow and memory-intensive. Modern architectures like **EfficientNet** or **Vision Transformers (ViT)** provide far better "accuracy-to-latency" ratios.