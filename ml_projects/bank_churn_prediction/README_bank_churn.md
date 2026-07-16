# Bank Churn Prediction — Neural Networks

A neural network classifier predicting whether a bank customer will churn within the next 6 months, iterated across several architectures to handle class imbalance and improve recall.

## Problem

Customer churn is expensive for banks — every customer who leaves is a loss the bank can't easily recover. This project builds a model to flag at-risk customers early enough for the bank to intervene, rather than reacting after the fact.

## Approach

- **Dataset**: ~10,000 bank customers, features covering credit score, geography, tenure, balance, product usage, and activity status
- **Evaluation metric**: Recall — for a churn-prevention use case, missing a customer who's about to leave (false negative) costs the bank far more than flagging a loyal customer as at-risk (false positive), which only costs a bit of unnecessary retention outreach
- **Class imbalance**: addressed with SMOTE (Synthetic Minority Over-sampling) after baseline models underperformed on the minority (churn) class
- **Model iteration**:

| Model | Configuration | Notes |
|---|---|---|
| 0 | SGD optimizer | Baseline |
| 1 | Adam optimizer | Faster convergence |
| 2 | Adam + Dropout | Regularization to reduce overfitting |
| 3 | SMOTE + Adam | Address class imbalance directly |
| 4 | SMOTE + Adam (deeper network) | Additional hidden layer |
| 5 | SMOTE + Adam + Dropout | Combine imbalance handling with regularization |

## Results

- **Best model**: Neural network with SMOTE-balanced data and the Adam optimizer, achieving the highest validation recall (~0.73) among all configurations tested
- This means the model correctly identifies roughly 3 out of 4 customers who are actually going to churn — a meaningful improvement over the unbalanced baseline

## Key takeaways

- Class imbalance was the single biggest lever — SMOTE meaningfully outperformed model architecture changes alone
- Recall, not accuracy, is the right optimization target when the cost of a missed positive case (a churned customer) is much higher than a false alarm
- The final model's confusion matrix and classification report were validated on a held-out test set, not just the validation set used for model selection

## Tech stack

Python · TensorFlow/Keras · scikit-learn · imbalanced-learn (SMOTE) · pandas · matplotlib/seaborn
