## 🏛 Architecture Overview

The VGG network (proposed by the Visual Geometry Group at Oxford) was a milestone in deep learning. Its primary contribution was demonstrating that **depth** is a critical component for high-performance visual recognition, achieved by using very small (**3x3**) convolution filters consistently throughout the network.

### 🍱 The VGG16 "Block" Strategy
![VGG](VGG16.png)
![VGG2](VGG16_2.png)

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

In the two notebook, several modern training techniques were implemented to ensure convergence and performance:

### ⚙️ Hyperparameters
- **Input Size**: 224x224x3
- **Batch Size**: 128
- **Learning Rate**: 0.01 (SGD with Momentum 0.9)
- **Dropout**: 0.5 (used in the classifier)
- **Weight Decay**: 5e-4

### 🛠 Modern Training Enhancements
- **AMP (Automatic Mixed Precision)**: Utilizing `torch.amp` for faster training and reduced memory footprint.
- **Learning Rate Scheduler**: `ReduceLROnPlateau` to dynamically adjust the LR when validation loss plateaus.
- **Early Stopping**: A custom class to prevent overfitting by monitoring validation loss (patience = 10).

---

## 🧪 Training Strategy: Paper vs. Mine

### The Original "Pre-training" Approach
In the original VGG paper, the authors explicitly explained that **pre-training** was used to prevent learning from stalling. This stalling is caused by the instability of gradients in very deep networks, which makes them highly sensitive to the initial weight values.

To solve this, the authors followed these steps:
1.  **Initial Training**: They trained **Configuration A** (a shallower 11-layer network), which was shallow enough to successfully converge using random initialization.
2.  **Weight Transfer**: The weights from this shallow model were then used to initialize specific layers in the deeper architectures (B, C, D, and E).
3.  **Random Initialization Alternative**: The authors later noted that pre-training becomes unnecessary if using the random initialization procedure of *Glorot & Bengio (2010)*, though they discovered this after their initial submission.

### My Implementation Approach
For learning purposes, I attempted to train **VGG16** directly from scratch without this complex pre-training step. And I trained only for 5 epochs just for testing.

*   **VGG16 (No Batch Norm)**: In the version close to the original paper (`VGG16_no_batch_normalisation.ipynb`), there was **no significant improvement** in learning progress even with **Random Initialization**. The model struggled to converge efficiently despite using modern initialization.
*   **The Fix: Batch Normalization**: To resolve this, I applied **Batch Normalization** (which was invented after VGG) in `VGG16_with_batch_normalisation.ipynb`. This successfully stabilized the training and significantly improved.

---

## 🔍 Interpretability with Grad-CAM

To understand **what** the model is looking at when making a prediction, we implement **Grad-CAM** (Gradient-weighted Class Activation Mapping).

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

## ⚠️ VGG16 Obsolete Now

Despite its historical importance, VGG16 is rarely used in production today for three main reasons:

1.  **Massive Parameter Weight**: VGG16 has **138 million parameters**, primarily due to the massive Fully Connected (FC) layers. A modern **ResNet-50** has only ~25M parameters while achieving higher accuracy.
2.  **Vanishing Gradients**: VGG16 lacks "Skip Connections." As networks get deeper than 20 layers without these connections, they become extremely difficult to train.
3.  **Computational Inefficiency**: It is slow and memory-intensive. Modern architectures like **EfficientNet** or **Vision Transformers (ViT)** provide far better "accuracy-to-latency" ratios.