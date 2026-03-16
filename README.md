# Deep Learning Implementations & Experiments

This repository serves as a comprehensive workspace for implementing, training, and experimenting with various Deep Learning architectures from scratch. The goal is to deeply understand the mechanics of state-of-the-art neural networks through hands-on implementation and fine-tuning.

## 📁 Models & Data

All trained models and test images are found in this **[Google Drive Folder](https://drive.google.com/drive/folders/1aENvxV8Xr01qDtnp2IOiO2IuzMRR8hBL?usp=drive_link)**.

## 🛠️ Tech Stack

- **Framework**: PyTorch and maybe TensorFlow later
- **Libraries**: NumPy, Matplotlib, OpenCV, Torchvision
- **Environment**: Jupyter Notebooks and Google Colab in VSCode

## 👁️ Experiment Tracking

All training runs are tracked with **[Weights & Biases (W&B)](https://wandb.ai/)**, which provides real-time logging of metrics (loss, accuracy, learning rate), interactive dashboards, and easy comparison across runs. Each notebook initializes a W&B run and logs training/validation/test metrics per epoch.

## 🖥️ Training Platform

To handle heavy training workloads, I leverage free GPU acceleration from:
- **[Kaggle](https://www.kaggle.com/)**: Primary training environment with ~30 hours of free GPU access per week.
- **[Lightning AI](https://lightning.ai/)**: Cloud-based compute using ~80 free hours for scalable experiments.

## 📖 Learning Resources

This repo follows academic and industry best practices, often referencing key literature such as Christopher Bishop's foundations and original research papers for each architecture.

---
*Maintained by Duong Tran*