# Safety Helmet Detection — CNN Image Classification

A binary image classifier that detects whether workers in industrial/construction settings are wearing safety helmets, built to support automated safety compliance monitoring at scale.

## Problem

Manual safety compliance monitoring doesn't scale well across large operations and is prone to human error. This project automates helmet-compliance detection from images, comparing a custom-built CNN against transfer learning with VGG-16 to find the best-performing approach.

## Approach

- **Dataset**: 631 images, roughly balanced between "with helmet" (311) and "without helmet" (320), covering varied lighting, angles, and worker activities
- **Preprocessing**: images normalized to 0–1 pixel range; grayscale conversion explored during EDA
- **Models compared**:

| Model | Architecture | Notes |
|---|---|---|
| 1 | Custom CNN (3 conv layers) | Built from scratch, no pretrained weights |
| 2 | VGG-16 (frozen, base) | Transfer learning baseline |
| 3 | VGG-16 (frozen) + FFNN head | Deeper classifier on top of frozen features |
| 4 | VGG-16 (frozen) + FFNN + data augmentation | Test whether augmentation improves generalization |

## Results

- VGG-16-based models outperformed the custom CNN, confirming the value of pretrained feature extractors even on a fairly small (631-image) dataset
- All models showed **low overfitting** — training and test performance stayed close, indicating good generalization within the current dataset's scope
- Data augmentation didn't meaningfully change test performance, suggesting the original dataset already had enough natural variation (lighting, angle, posture) to cover what augmentation would otherwise add

## Recommendation

Proceed with the VGG-16 + FFNN + data augmentation configuration for deployment — it gave the most consistent evaluation metrics across train/validation/test splits, even though augmentation itself didn't move the needle much on this dataset; it's a safer choice if the model is later deployed on more varied real-world images than what's in the training set.

## Tech stack

Python · TensorFlow/Keras · VGG-16 · OpenCV · scikit-learn · matplotlib/seaborn
