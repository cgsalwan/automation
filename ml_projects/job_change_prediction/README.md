# Job Change Prediction — Neural Network Optimization

A binary classification project predicting whether a data-science trainee is likely to look for a new job after completing training, built while comparing multiple neural network configurations to understand what actually improves generalization.

## Problem

A training provider wants to know, based on data collected at signup, which trainees are likely to be job-hunting versus staying — useful for planning hiring pipelines, retention outreach, and training investment. The target is imbalanced (~25% positive class), which shapes the evaluation strategy below.

## Approach

- **Preprocessing**: missing-value imputation (mode for categoricals), one-hot encoding, feature scaling, and binning of the experience variable into 5 groups
- **Class imbalance**: handled via class weighting rather than resampling, keeping the full dataset intact
- **Evaluation metric**: F1 score (weighted) chosen over raw accuracy, since both false positives and false negatives carry real cost for the business
- **Model iteration**: 7 neural network configurations were trained and compared, isolating one change at a time:

| Model | Change | Purpose |
|---|---|---|
| 0 | Baseline (SGD) | Establish reference performance |
| 1 | + Momentum | Test whether learning speed was the bottleneck |
| 2 | Adam optimizer | Adaptive learning rate |
| 3 | + Dropout | Reduce train/validation gap |
| 4 | + Batch Normalization | Stabilize training |
| 5 | Dropout + BatchNorm | Combine both regularization techniques |
| 6 | + He initialization | Better weight init for ReLU layers |

The final model (Adam + Dropout + He initialization) was selected based on validation F1 score and the smallest train/validation gap, indicating the best generalization among the configurations tested.

## Results

- Final weighted F1 score on held-out test data: **~0.74**
- Consistent performance across train/validation/test, indicating the model isn't overfitting

## Key takeaways

- Employees with non-relevant prior experience, less than 3 years of experience, or who've never switched jobs before are more likely to be job-hunting
- Company type and gender showed little to no relationship with job-change likelihood in this dataset
- The dataset itself is ~90% male, a limitation worth noting for any real-world deployment of this kind of model

## Tech stack

Python · TensorFlow/Keras · scikit-learn · pandas · matplotlib/seaborn

## Notes

This project was completed as part of a postgraduate AI/ML program (UT Austin, McCombs School of Business).
