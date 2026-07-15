# Monkey Species Classification — Convolutional Neural Networks

A CNN-based image classifier that identifies 10 endangered monkey species from image data, built while comparing transfer learning strategies to find the best trade-off between accuracy and overfitting.

## Problem

Manually sorting large volumes of wildlife image data by species is slow and error-prone, especially across visually similar species — a real bottleneck for conservation research that depends on tracking population trends from image data. This project automates that classification step.

## Approach

- **Dataset**: 1,342 RGB images (128×128×3) across 10 species, reasonably balanced (~135 images/class)
- **Preprocessing**: resized to 64×64 for computational efficiency, normalized pixel values to 0–1, labels one-hot encoded
- **Transfer learning**: used VGG-16 (pretrained on ImageNet) as a frozen feature extractor, then compared 3 configurations built on top of it:

| Model | Architecture | Purpose |
|---|---|---|
| 1 | VGG-16 (frozen) + Flatten + Dense output | Baseline transfer learning |
| 2 | VGG-16 (frozen) + FFNN (256 → dropout → 32) + Dense output | Test whether a deeper classifier head improves results |
| 3 | Model 2 + data augmentation (rotation, shift, shear, zoom) | Test regularization against overfitting |

## Results

- **Model 1 (VGG-16 base)**: strong training performance (F1 ~0.92) but a large train/validation gap (val F1 ~0.62) — clear overfitting
- **Model 2 (+ FFNN head)**: best overall performance, validation F1 ~0.62 with better generalization than the base model
- **Model 3 (+ data augmentation)**: lower raw performance (~0.62 train F1) but no overfitting, confirming augmentation as an effective regularizer even though it wasn't the top performer here
- **Final model**: Model 2 (VGG-16 + FFNN), selected for the best balance of accuracy and generalization

## Key takeaways

- Pretrained ImageNet features transfer reasonably well to a specialized, small dataset like this one, but a frozen base alone overfits
- A deeper classifier head on top of frozen convolutional features meaningfully improved results over the bare baseline
- Data augmentation is a viable next step to close the remaining generalization gap, and could likely be tuned further (stronger augmentation, unfreezing some VGG-16 layers, training on the original 128×128 resolution)

## Tech stack

Python · TensorFlow/Keras · VGG-16 · OpenCV · scikit-learn · matplotlib/seaborn
