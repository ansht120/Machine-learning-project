# Image Classification — Handwritten Digit Recognition

> CNN built with TensorFlow/Keras that recognises handwritten digits (0–9) from the MNIST dataset.

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)

**Result: 98.83% test accuracy**

---

## Problem

Recognising handwritten digits is the foundational computer-vision task behind postal code sorting, cheque digitisation, and form processing. The challenge is variation — no two people write a "7" identically, yet the model must generalise across all of them.

## Dataset

**MNIST** — 60,000 training and 10,000 test images, 28×28 pixels, greyscale. Loaded directly through `keras.datasets`, so no manual download is needed.

## Approach

### Preprocessing

1. **Normalisation** — pixel values scaled from `[0, 255]` to `[0, 1]`. Neural networks converge faster and more stably when inputs share a small, consistent range.
2. **Reshaping** — a channel dimension is added (`28×28` → `28×28×1`) because `Conv2D` expects a channel axis even for greyscale input.

### Architecture

```
Input (28×28×1)
  ↓ Conv2D(32, 3×3, ReLU)   →  MaxPooling2D(2×2)
  ↓ Conv2D(64, 3×3, ReLU)   →  MaxPooling2D(2×2)
  ↓ Conv2D(64, 3×3, ReLU)
  ↓ Flatten
  ↓ Dense(64, ReLU)
  ↓ Dense(10, Softmax)
```

**Design reasoning:**

- **Filters increase (32 → 64)** as spatial dimensions shrink. Early layers detect simple edges and need few filters; deeper layers combine those into complex shapes and need more.
- **MaxPooling** halves the spatial size after each of the first two blocks, cutting computation and providing small translation invariance — a digit shifted a few pixels still classifies correctly.
- **Softmax output over 10 units** converts raw scores to a probability distribution across the digit classes.

### Training

| Setting | Value |
|---------|-------|
| Optimiser | Adam |
| Loss | Sparse categorical cross-entropy |
| Epochs | 5 |

*Sparse* categorical cross-entropy is used because labels are integers (`0`–`9`) rather than one-hot vectors — it avoids an unnecessary encoding step.

---

## Results

### Training progression

| Epoch | Accuracy | Loss |
|-------|----------|------|
| 1 | 88.98% | 0.3449 |
| 2 | 98.46% | 0.0504 |
| 3 | 99.01% | 0.0332 |
| 4 | 99.21% | 0.0253 |
| 5 | 99.43% | 0.0185 |

### Final evaluation

| Metric | Value |
|--------|-------|
| **Test accuracy** | **98.83%** |
| Test loss | 0.0491 |

> **Interpretation:** the gap between training accuracy (99.43%) and test accuracy (98.83%) is just 0.6 percentage points — narrow enough to indicate the model learned generalisable features rather than memorising the training set. The jump from 88.98% to 98.46% between epochs 1 and 2 is characteristic of CNNs on MNIST: the convolutional filters converge on useful edge detectors very quickly.

The trained model is exported to `mnist_model.h5` for reuse without retraining.

---

## Running It

```bash
pip install tensorflow numpy matplotlib
jupyter notebook imageclassification.ipynb
```

No dataset download required — MNIST is fetched automatically on first run. Training takes roughly 5 minutes on CPU.

---

## Possible Improvements

- **Dropout layers** between dense layers to regularise further
- **Data augmentation** (small rotations, shifts) to improve robustness to unusual handwriting
- **Extend to harder datasets** — Fashion-MNIST or CIFAR-10, where 98% is far from trivial

---

[⬅ Back to all ML projects](../README.md)
