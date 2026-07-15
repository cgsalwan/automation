# HR Employee Attrition Prediction — Ensemble Methods

A classification project predicting which employees are at risk of attrition, so retention efforts can be targeted rather than applied broadly across an entire workforce.

## Problem

Retention initiatives are expensive when applied company-wide. This project identifies the key factors that drive employee attrition and builds a model to flag at-risk employees, so incentives and retention efforts can be targeted where they matter most.

## Approach

- **Dataset**: 2,940 employee records, 35 features (demographics, compensation, satisfaction ratings, tenure) — target class is imbalanced (~16% attrition)
- **Evaluation metric**: Recall, prioritized over accuracy — for this business problem, failing to flag an employee who actually leaves (false negative) is far costlier than a false alarm
- **Models compared**:

| Model | Notes |
|---|---|
| Decision Tree | Baseline, with class weighting to counter imbalance |
| Bagging Classifier | Overfit on training data, weak test recall |
| Bagging + weighted Decision Tree | Strong accuracy, still weak generalization on recall |
| Random Forest | Good accuracy/precision, weak test recall |
| Random Forest (weighted) | No meaningful improvement over unweighted |
| Tuned Decision Tree (GridSearchCV) | Reduced overfitting, but recall dropped |
| Tuned Bagging (GridSearchCV) | Recall improved, but accuracy/precision dropped sharply |
| Tuned Random Forest (GridSearchCV) | Comparable to untuned Random Forest |

## Key findings

- **Monthly Income**, **OverTime**, and **Age** are the strongest drivers of attrition
- Employees who work overtime, earn less, or are younger are meaningfully more likely to leave
- Distance from home also correlates with attrition — longer commutes, higher attrition risk
- Every model tested showed a real trade-off between recall and precision/accuracy — none dominated across all metrics, which is a realistic and important finding for a business deciding which model to deploy

## Business recommendations

- Benchmark and adjust compensation for consistently underpaid employees, since income was the top predictor
- Provide additional incentives or workload support for employees who regularly work overtime
- Focus retention/engagement efforts on younger employees and those with longer commutes

## Tech stack

Python · scikit-learn (Decision Trees, Bagging, Random Forest, GridSearchCV) · pandas · matplotlib/seaborn
