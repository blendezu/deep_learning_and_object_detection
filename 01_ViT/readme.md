# 01 — Vision Transformer (ViT)

This module explores the **Vision Transformer (ViT)** architecture end-to-end, starting from building one from scratch, all the way through pre-training and fine-tuning on real-world datasets.

> 📖 Reference paper: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929) (Dosovitskiy et al., 2021)

---

## 📋 Contents

| Notebook | Description |
|---|---|
| `simple_ViT_from_scratch.ipynb` | A simple Script for understanding Vision Transformer |
| `pre_train_Mini-ViT_from_scratch.ipynb` | Pre-train a Mini-ViT (~3.2M params) on CIFAR-100 |
| `fine_tuning_Mini-ViT_self_pretrained.ipynb` | Fine-tune the self-pretrained Mini-ViT on Food-101 (Pizza vs. Sushi) |
| `fine_tuning_ViT_google_pretrained.ipynb` | Fine-tune Google's ViT-B/16 on a Fake vs. Real Face dataset |
| `fine_tuning_ViT_google_pretrained_more_aug.ipynb` | Same as above, with stronger data augmentation |
| `inference_pizza_sushi.ipynb` | Run inference with the fine-tuned Pizza vs. Sushi model |
| `inference_fake_real.ipynb` | Run inference with the fine-tuned Fake vs. Real Face model |

---

## 🔬 Stage 1 — Simple ViT from Scratch

**Notebook:** `simple_ViT_from_scratch.ipynb`

A clean, heavily commented implementation of ViT built entirely from PyTorch primitives. The goal is to understand every component of the architecture before moving to training at scale.

### Architecture

The model implements the full ViT pipeline:

```
Image → Patch Embedding → [CLS] Token + Positional Embedding
      → Transformer Encoder (×L layers)
      → LayerNorm
      → MLP Head → Class Logits
```

**Key components built from scratch:**

- **`PatchEmbedding`** — Splits the image into non-overlapping patches using a `Conv2d` with `kernel_size = stride = patch_size`, then prepends a learnable `[CLS]` token and adds learnable positional embeddings.
- **`MLP`** — A two-layer feed-forward block (`Linear → GELU → Dropout → Linear → Dropout`).
- **`TransformerEncoderLayer`** — One encoder block with Pre-LN Multi-Head Self-Attention and an MLP sub-layer, both with residual connections.
- **`VisionTransformer`** — Stacks `L` encoder layers and reads the `[CLS]` token for classification.

### Dataset & Training

| Setting | Value |
|---|---|
| Dataset | CIFAR-10 (10 classes) |
| Image size | 32 × 32 |
| Patch size | 4 × 4 (64 patches) |
| Embed dim | 256 |
| Attention heads | 8 |
| Encoder depth | 6 |
| MLP dim | 512 |
| Dropout | 0.1 |
| Optimizer | AdamW (lr = 3e-4, weight decay = 1e-4) |
| Scheduler | CosineAnnealingLR |
| Epochs | 30 |

**Result:** ~77.5% test accuracy on CIFAR-10.

---

## 🔬 Stage 2 — Pre-training a Mini-ViT from Scratch

**Notebook:** `pre_train_Mini-ViT_from_scratch.ipynb`

Trains a more capable Mini-ViT (~3.2M parameters) from scratch on the harder **CIFAR-100** dataset (100 classes). This produces a general-purpose backbone that is later reused for fine-tuning.

### Key Differences from Stage 1

- **Dataset:** CIFAR-100 instead of CIFAR-10 — much harder (100 classes, 600 images per class).
- **Augmentation:** `AutoAugment` (CIFAR10 policy) + `RandomCrop` + `RandomHorizontalFlip` + `RandomErasing` for strong regularization.
- **MLP Head:** Upgraded from a single linear layer to a non-linear classification head (`LayerNorm -> Linear -> GELU -> Dropout -> Linear`) to enable complex feature mapping, as described in the original ViT paper.
- **Label Smoothing:** `CrossEntropyLoss(label_smoothing=0.1)` to prevent overconfidence.
- **Optimizer:** AdamW with weight decay of 0.05.
- **Scheduler:** `OneCycleLR` — starts low, warms up to `3e-4`, then anneals to near zero.
- **Gradient Clipping:** `clip_grad_norm_` (max_norm = 1.0) for training stability.
- **Early Stopping:** Monitors validation loss with patience = 10.
- **Experiment Tracking:** [Weights & Biases](https://api.wandb.ai/links/duongsemailforeverything-/e7up3s1h)

### Attention Rollout Visualization

> 📖 Reference paper: [Quantifying Attention Flow in Transformers](https://arxiv.org/pdf/2005.00928)

After training, **Attention Rollout** is used to visualize what the model "looks at" when making predictions. It works by recursively multiplying attention weights across all encoder layers (accounting for residual connections), producing a heatmap that highlights the most salient image regions for the `[CLS]` token.

### Model Architecture

```
ViT(
  embed = EmbeddedPatches(Conv2d(3, 256, 4×4))
  encoder = Sequential(VisionEncoder × 6)
  mlp_head = ClassHead(
    norm = LayerNorm(256)
    layer1 = Linear(256 → 512)
    act = GELU()
    layer2 = Linear(512 → 100)
  )
)
```

The saved pre-trained weights (`pre_trained_ViT_CIFAR100.pth`) are used as the starting point for Stage 3.

---

## 🔬 Stage 3 — Fine-tuning the Self-Pre-trained ViT

**Notebook:** `fine_tuning_Mini-ViT_self_pretrained.ipynb`

Demonstrates how to fine-tune **your own custom pre-trained ViT** (from Stage 2) for a completely different task at a higher resolution.

### Task

**Binary classification:** Pizza 🍕 vs. Sushi 🍣 — using the **Food-101** dataset (torchvision).

### Key Challenges & Solutions

**1. Resolution Change (32×32 → 64×64)**

The pre-trained model was trained on 32×32 images (64 patches). At 64×64 with the same patch size of 4×4, the number of patches becomes 256. Since the positional embeddings are learnable parameters tied to a specific sequence length, they must be adapted.

**Solution — Positional Embedding Interpolation:**
```
old shape: (1, 65, 256)   [64 patches + 1 CLS]
new shape: (1, 257, 256)  [256 patches + 1 CLS]
```
The `[CLS]` token embedding is kept as-is. The grid tokens (64 → 8×8) are reshaped into a 2D grid, bicubically interpolated to 16×16, then flattened back into a sequence of 256 tokens.

**2. Class Mismatch (100 → 2 classes)**

The old `mlp_head` (Linear 256→100) is discarded. A new `mlp_head` (Linear 256→2) is zero-initialized, as described in the original ViT paper.

### Training Details

| Setting | Value |
|---|---|
| Dataset | Food-101 (Pizza vs. Sushi subset) |
| Train / Val / Test | ~1200 / 300 / 500 images |
| Image size | 64 × 64 |
| Optimizer | AdamW (lr = 1e-4, weight decay = 1e-4) |
| Mixed Precision | `torch.cuda.amp` (FP16 + GradScaler) |
| Early Stopping | Patience = 5 |
| Experiment Tracking | Weights & Biases |

**Result:** ~95% test accuracy (best checkpoint).

---

## 🔬 Stage 4 — Fine-tuning Google's Pre-trained ViT-B/16

**Notebook:** `fine_tuning_ViT_google_pretrained.ipynb`  
**Notebook (more augmentation):** `fine_tuning_ViT_google_pretrained_more_aug.ipynb`

Uses `torchvision.models.vit_b_16` with **ImageNet pre-trained weights** (ViT-B/16), fine-tuned on a real-world binary classification task.

### Task

**Binary classification:** Fake vs. Real Faces — using the [DeepDetect-2025 dataset](https://www.kaggle.com/datasets/ayushmandatta1/deepdetect-2025) from Kaggle.

> This is a practical use case for detecting AI-generated faces, a relevant challenge in the era of deep fakes.

### Key Details

**Positional Embedding Interpolation (224 → 256 resolution)**

The model is loaded with `image_size=256` so it supports 256×256 inputs. However, the pretrained weights encode positional embeddings for 14×14 = 196 patches (from 224×224 at patch size 16). At 256×256, the grid becomes 16×16 = 256 patches.

```
old grid: 14×14 (196 patches)  →  new grid: 16×16 (256 patches)
Interpolation mode: bicubic
```

**Classification Head Replacement**

The original `model.heads.head` (Linear 768→1000) is replaced with a new `Linear(768 → 2)`, zero-initialized as per the ViT paper.

### Training Details

| Setting | Value |
|---|---|
| Dataset | DeepDetect-2025 (Fake vs. Real Faces) |
| Image size | 256 × 256 |
| Batch size | 32 |
| Optimizer | AdamW (lr = 1e-4, weight decay = 0.01) |
| Scheduler | OneCycleLR (warmup = 10% of steps) |
| Loss | CrossEntropyLoss (label_smoothing = 0.1) |
| Mixed Precision | `torch.cuda.amp` |
| Early Stopping | Patience = 3 |
| Experiment Tracking | [Weights & Biases](https://wandb.ai/duongsemailforeverything-/FT-ViT-Google-pretrained) |

**Result:**
- **Baseline (`fine_tuning_ViT_google_pretrained.ipynb`):** Val accuracy ~99.6%, but test accuracy is volatile (78–93%) — the test set was collected from different generators/environments than the training distribution.
- **Improved (`fine_tuning_ViT_google_pretrained_more_aug.ipynb`):** By applying stronger data augmentation (**RandAugment**, Gaussian Blur, Random Perspective, and Color Jitter), the model generalized significantly better. The test accuracy stabilized and reached **~96%**.

---

## 🔬 Stage 5 — Inference

**Notebooks:** `inference_pizza_sushi.ipynb`, `inference_fake_real.ipynb`

These notebooks load the best fine-tuned checkpoints and run inference on new images, demonstrating the full deployment pipeline:

- Load saved `.pth` weights
- Apply the correct preprocessing transforms
- Run forward pass and decode predictions

---

## 📊 Results Summary

| Stage | Task | Dataset | Best Test Acc |
|---|---|---|---|
| Simple ViT (scratch) | 10-class classification | CIFAR-10 | ~77.5% |
| Mini-ViT Pre-training | 100-class classification | CIFAR-100 | ~55–60% |
| Fine-tune (self-pretrained) | Pizza vs. Sushi | Food-101 subset | ~95% |
| Fine-tune (Google ViT-B/16) | Fake vs. Real Faces | DeepDetect-2025 | **~96%** |
