# Loan Default Prediction — Boosting Ensembles

A classification project comparing three boosting algorithms to predict which loan applicants are likely to default, optimized for recall since missing an actual defaulter is costlier to a bank than a false alarm.

## Problem

Banks need to flag high-risk applicants before approving a loan, not after. Using applicant financial data — checking account balance, credit history, loan purpose, loan amount, and related features — this project builds and tunes three boosting classifiers to predict default risk.

## Approach

- **Dataset**: German credit data, applicant financial/demographic features with a binary default flag
- **Preprocessing**: categorical encoding (ordinal mapping + one-hot encoding), stratified train/test split to preserve class balance
- **Evaluation metric**: Recall — a missed defaulter (false negative) costs the bank far more than a false alarm (false positive)
- **Models compared**, each with default parameters and then GridSearchCV-tuned:

| Model | Default params | Tuned |
|---|---|---|
| AdaBoost | Best test accuracy of the three, but weak recall | Overfits; low test recall — not good enough for this use case |
| Gradient Boosting | Lowest accuracy and recall of the three | Marginal improvement; starts overfitting on recall |
| XGBoost | — | **Best result** — recall improved significantly, doesn't overfit, generalizes well |

## Key findings

- **XGBoost (tuned) was the clear winner** — it traded a modest accuracy drop for a significant recall improvement, catching most actual defaulters without overfitting
- **Loan amount** was the top predictor for both AdaBoost and Gradient Boosting; **checking account balance** was the top predictor for XGBoost — a reminder that feature importance isn't always consistent across model types
- Recall isn't the whole picture: I worked through a numerical example showing that over-optimizing for recall (catching every defaulter) can still lose the bank money overall if it comes at the cost of rejecting too many good applicants — the real objective is a cost function balancing both error types, not recall in isolation

## Tech stack

Python · scikit-learn (AdaBoost, GridSearchCV) · XGBoost · pandas · matplotlib/seaborn
