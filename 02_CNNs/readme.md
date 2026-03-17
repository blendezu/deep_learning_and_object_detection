# 🖼️ CNN Implementations

This directory is dedicated to the implementation of the most important Convolutional Neural Network (CNN) architectures in history. Each folder contains a self-contained implementation and sometimes a notebook explaining the key concepts and architecture details.

### CNN Timeline

| Year | Model | Status | Improvement | Usage |
| :--- | :--- | :--- | :--- | :--- |
| 1998 | [LeNet-5](./01_LeNet5) | **Historical** | First successful application of CNNs | Handwritten digit recognition (checks/postal) |
| 2012 | [AlexNet](./02_AlexNet) | **Obsolete** | Deeper network, ReLU, Dropout, and GPU acceleration | Large-scale image classification (ImageNet) |
| 2013 | [ZFNet](./03_ZFNet) | **Obsolete** | Refined AlexNet hyperparameters via visualization | Research and visualization of feature maps |
| 2014 | [VGG](./04_VGG) | **Obsolete** | Standardized small $3 \times 3$ filters for deeper layers | Feature extraction and transfer learning |
| 2014 | [GoogLeNet](./05_GoogLeNet) | **Outdated** | Inception modules for multiscale processing | High performance with low computational cost |
| 2015 | [ResNet](./06_ResNet) | **Standard** | Residual connections to solve vanishing gradients | Backbone for almost all CV tasks today |
| 2016 | [DenseNet](./07_DenseNet) | **Specialized** | Dense connections for maximum feature reuse | Medical imaging and tasks needing high efficiency |
| 2017 | [MobileNet](./08_MobileNet) | **Relevant** | Depthwise separable convolutions for mobile efficiency | Mobile apps, IoT, and real-time edge devices |
| 2019 | [EfficientNet](./09_EfficientNet) | **Highly Relevant** | Compound scaling (width, depth, resolution) | SOTA classification with high efficiency |
| 2022 | [ConvNeXt](./10_ConvNeXt) | **Modern** | Modernizing CNNs with Transformer-inspired design | Modern high-performance vision backbones |