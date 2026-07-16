# Visa Approval Prediction — Ensemble Methods

A classification project predicting whether a US work visa (permanent labor certification) application is likely to be certified or denied, built to help streamline case triage by flagging applications likely to need additional review.

## Problem

Employers sponsoring foreign workers for US employment must go through a labor certification process administered by the Office of Foreign Labor Certification (OFLC). Reviewing every application manually is slow; a model that predicts likely outcomes upfront can help prioritize case review and set expectations earlier in the process.

## Approach

- **Dataset**: ~25,000 visa applications with employer, employee, and job-posting features (education, job experience, region of employment, prevailing wage, employer size, etc.)
- **EDA**: examined how education level, job experience, continent of origin, and prevailing wage relate to case outcome
- **Class imbalance handling**: compared three data strategies — original (imbalanced) data, SMOTE-oversampled data, and undersampled data — across all models before tuning
- **Models compared**: Bagging, Random Forest, Gradient Boosting, AdaBoost, and XGBoost, with cross-validated F1 score as the initial screening metric, followed by RandomizedSearchCV hyperparameter tuning on the strongest candidates
- **Evaluation**: models were tuned and validated separately, then the single best-performing model was evaluated once on a held-out test set to get an unbiased final performance estimate

## Key findings

- **XGBoost was the best-performing model** on validation data after tuning, and was selected as the final model for test-set evaluation
- Prevailing wage, education level, and job experience were consistently among the most predictive features across models
- Comparing original, oversampled, and undersampled data directly (rather than assuming one approach is best) surfaced meaningful differences in which model performed best under each sampling strategy — worth doing rather than defaulting to one imbalance-handling technique

## Tech stack

Python · scikit-learn (Bagging, Random Forest, Gradient Boosting, AdaBoost, RandomizedSearchCV) · XGBoost · imbalanced-learn (SMOTE, RandomUnderSampler) · pandas · matplotlib/seaborn
