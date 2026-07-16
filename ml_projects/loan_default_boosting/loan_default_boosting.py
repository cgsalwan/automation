"""
Loan Default Prediction — Boosting Ensembles

Compares AdaBoost, Gradient Boosting, and XGBoost (default and
hyperparameter-tuned) to predict loan default risk, optimized for recall.
"""

# # Loan Default Prediction — Boosting Ensembles
# 
# *Comparing AdaBoost, Gradient Boosting, and XGBoost — with hyperparameter tuning — to predict loan default risk, optimized for recall.*

# ## Problem
# 
# Banks need to identify loan applicants who are likely to default before approving a loan, not after. Using applicant data — checking account balance, credit history, loan purpose, loan amount, and related features — this project builds and compares three boosting-based classifiers to predict default risk, with an emphasis on catching as many actual defaulters as possible.

# ## Import Libraries

# ## Import Libraries

import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
# %matplotlib inline  (Jupyter magic, not needed as a script)
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
#To install xgboost library use - !pip install xgboost
from xgboost import XGBClassifier

# The dataset is a standard German credit dataset containing applicant financial history and a default flag.

creditData = pd.read_csv("credit.csv")

creditData = pd.read_csv("credit.csv")
creditData.head(10) #several missing values!

creditData.shape

creditData['default'].value_counts()

creditData.describe()

creditData.info()  # many columns are of type object i.e. strings. These need to be converted to ordinal type

# Converting columns with an 'object' datatype into categorical variables:

for feature in creditData.columns: # Loop through all columns in the dataframe
    if creditData[feature].dtype == 'object': # Only apply for columns with categorical strings
        creditData[feature] = pd.Categorical(creditData[feature])# Replace strings with an integer
creditData.head(10)

print(creditData.checking_balance.value_counts())
print(creditData.credit_history.value_counts())
print(creditData.purpose.value_counts())
print(creditData.savings_balance.value_counts())
print(creditData.employment_duration.value_counts())
print(creditData.other_credit.value_counts())
print(creditData.housing.value_counts())
print(creditData.job.value_counts())
print(creditData.phone.value_counts())

replaceStruct = {
                "checking_balance":     {"< 0 DM": 1, "1 - 200 DM": 2 ,"> 200 DM": 3 ,"unknown":-1},
                "credit_history": {"critical": 1, "poor":2 , "good": 3, "very good": 4,"perfect": 5},
                 "savings_balance": {"< 100 DM": 1, "100 - 500 DM":2 , "500 - 1000 DM": 3, "> 1000 DM": 4,"unknown": -1},
                 "employment_duration":     {"unemployed": 1, "< 1 year": 2 ,"1 - 4 years": 3 ,"4 - 7 years": 4 ,"> 7 years": 5},
                "phone":     {"no": 1, "yes": 2 },
                #"job":     {"unemployed": 1, "unskilled": 2, "skilled": 3, "management": 4 },
                "default":     {"no": 0, "yes": 1 }
                    }
oneHotCols=["purpose","housing","other_credit","job", "checking_balance",
            "credit_history", "savings_balance", "employment_duration", "phone"]

creditData=creditData.replace(replaceStruct)
creditData=pd.get_dummies(creditData, columns=oneHotCols)
creditData.head(10)

creditData.info()

# ## Split the data into train and test sets

# Since the target class (default) is imbalanced, I used stratified sampling in the train/test split (via the `stratify` parameter) to keep the class ratio consistent across both sets.

X = creditData.drop("default" , axis=1)
y = creditData.pop("default")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=1,stratify=y)

# Reusing the confusion matrix and scoring helper functions defined below for evaluating each model.

## Function to create confusion matrix
def make_confusion_matrix(model,y_actual,labels=[1, 0]):
    '''
    model : classifier to predict values of X
    y_actual : ground truth

    '''
    y_predict = model.predict(X_test)
    cm=metrics.confusion_matrix( y_actual, y_predict, labels=[0, 1])
    df_cm = pd.DataFrame(cm, index = [i for i in ["Actual - No","Actual - Yes"]],
                  columns = [i for i in ['Predicted - No','Predicted - Yes']])
    group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cm.flatten()/np.sum(cm)]
    labels = [f"{v1}\n{v2}" for v1, v2 in
              zip(group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=labels,fmt='')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

##  Function to calculate different metric scores of the model - Accuracy, Recall and Precision
def get_metrics_score(model,flag=True):
    '''
    model : classifier to predict values of X

    '''
    # defining an empty list to store train and test results
    score_list=[]

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    train_acc = model.score(X_train,y_train)
    test_acc = model.score(X_test,y_test)

    train_recall = metrics.recall_score(y_train,pred_train)
    test_recall = metrics.recall_score(y_test,pred_test)

    train_precision = metrics.precision_score(y_train,pred_train)
    test_precision = metrics.precision_score(y_test,pred_test)

    score_list.extend((train_acc,test_acc,train_recall,test_recall,train_precision,test_precision))

    # If the flag is set to True then only the following print statements will be dispayed. The default value is set to True.
    if flag == True:
        print("Accuracy on training set : ",model.score(X_train,y_train))
        print("Accuracy on test set : ",model.score(X_test,y_test))
        print("Recall on training set : ",metrics.recall_score(y_train,pred_train))
        print("Recall on test set : ",metrics.recall_score(y_test,pred_test))
        print("Precision on training set : ",metrics.precision_score(y_train,pred_train))
        print("Precision on test set : ",metrics.precision_score(y_test,pred_test))

    return score_list # returning the list with train and test scores

# ## Building the models
# 
# I built three boosting models — AdaBoost, Gradient Boosting, and XGBoost — first with default parameters, then tuned via GridSearchCV. I tracked Accuracy, Precision, and Recall throughout, but **recall is the metric I optimized for**: a high recall means fewer false negatives, i.e. fewer actual defaulters slipping through as predicted non-defaulters — which is the costlier mistake for a bank to make.

# ### AdaBoost Classifier

abc = AdaBoostClassifier(random_state=1)
abc.fit(X_train,y_train)

#Using above defined function to get accuracy, recall and precision on train and test set
abc_score=get_metrics_score(abc)

make_confusion_matrix(abc,y_test)

# ### Gradient Boosting Classifier

gbc = GradientBoostingClassifier(random_state=1)
gbc.fit(X_train,y_train)

#Using above defined function to get accuracy, recall and precision on train and test set
gbc_score=get_metrics_score(gbc)

make_confusion_matrix(gbc,y_test)

# ### XGBoost Classifier

xgb = XGBClassifier(random_state=1,eval_metric='logloss')
xgb.fit(X_train, y_train)

#Using above defined function to get accuracy, recall and precision on train and test set
xgb_score=get_metrics_score(xgb)

make_confusion_matrix(xgb,y_test)

# **With default parameters:**
# - AdaBoost had the best test accuracy of the three models
# - Gradient Boosting had the lowest test accuracy and test recall

# ## Hyperparameter Tuning
# 
# ### AdaBoost Classifier

# AdaBoost is a meta-estimator that fits a classifier on the data, then fits successive copies of it with the weights of misclassified instances increased, so later classifiers focus more on the harder cases. Key hyperparameters I tuned:
# - `base_estimator`: the estimator the ensemble is built from (default: a decision tree with `max_depth=1`)
# - `n_estimators`: max number of estimators before boosting stops (default: 50)
# - `learning_rate`: shrinks each classifier's contribution — trades off against `n_estimators`

# Choose the type of classifier.
abc_tuned = AdaBoostClassifier(random_state=1)

# Grid of parameters to choose from
## add from article
parameters = {
    #Let's try different max_depth for base_estimator
    "estimator":[DecisionTreeClassifier(max_depth=1, random_state=1),DecisionTreeClassifier(max_depth=2, random_state=1),DecisionTreeClassifier(max_depth=3, random_state=1)],
    "n_estimators": np.arange(10,110,10),
    "learning_rate":np.arange(0.1,2,0.1)
}

# Type of scoring used to compare parameter combinations
acc_scorer = metrics.make_scorer(metrics.recall_score)

# Run the grid search
grid_obj = GridSearchCV(abc_tuned, parameters, scoring=acc_scorer,cv=5)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
abc_tuned = grid_obj.best_estimator_

# Fit the best algorithm to the data.
abc_tuned.fit(X_train, y_train)

#Using above defined function to get accuracy, recall and precision on train and test set
abc_tuned_score=get_metrics_score(abc_tuned)

make_confusion_matrix(abc_tuned,y_test)

# **My takeaways on the tuned AdaBoost model:**
# - It's overfitting — train accuracy is noticeably higher than test accuracy
# - Test recall is low, meaning it's not reliably catching actual defaulters — not good enough for this use case

importances = abc_tuned.feature_importances_
indices = np.argsort(importances)
feature_names = list(X.columns)

plt.figure(figsize=(12,12))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='violet', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# Loan amount came out as the most important feature in the tuned AdaBoost model.

# ### Gradient Boosting Classifier

# Most of the hyperparameters here mirror Random Forest's. One notable difference: `init`, which sets the estimator used for initial predictions (defaults to a dummy estimator predicting class priors if not set). Gradient Boosting also has no `class_weight` parameter, unlike some other classifiers.

# **I tried using AdaBoost as the initial estimator (`init`) to see if it improved results:**

gbc_init = GradientBoostingClassifier(init=AdaBoostClassifier(random_state=1),random_state=1)
gbc_init.fit(X_train,y_train)

#Using above defined function to get accuracy, recall and precision on train and test set
gbc_init_score=get_metrics_score(gbc_init)

# **Compared to the default-parameter model:**
# - Test accuracy and test recall both improved slightly
# - Since this improved results, I used `init=AdaBoostClassifier()` going into the tuning step

# Choose the type of classifier.
gbc_tuned = GradientBoostingClassifier(init=AdaBoostClassifier(random_state=1),random_state=1)

# Grid of parameters to choose from
## add from article
parameters = {
    "n_estimators": [100,150,200,250],
    "subsample":[0.8,0.9,1],
    "max_features":[0.7,0.8,0.9,1]
}

# Type of scoring used to compare parameter combinations
acc_scorer = metrics.make_scorer(metrics.recall_score)

# Run the grid search
grid_obj = GridSearchCV(gbc_tuned, parameters, scoring=acc_scorer,cv=5)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
gbc_tuned = grid_obj.best_estimator_

# Fit the best algorithm to the data.
gbc_tuned.fit(X_train, y_train)

#Using above defined function to get accuracy, recall and precision on train and test set
gbc_tuned_score=get_metrics_score(gbc_tuned)

make_confusion_matrix(gbc_tuned,y_test)

# **My takeaways on the tuned Gradient Boosting model:**
# - Performance didn't improve by much over the untuned version
# - It started overfitting on recall specifically
# - It's better at identifying non-defaulters than defaulters — the opposite of what this problem needs

importances = gbc_tuned.feature_importances_
indices = np.argsort(importances)
feature_names = list(X.columns)

plt.figure(figsize=(12,12))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='violet', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# Loan amount was again the most important feature, followed by loan duration and applicant age.

# ### XGBoost Classifier

# XGBoost has a large hyperparameter surface — full documentation [here](https://xgboost.readthedocs.io/en/latest/parameter.html#general-parameters). The ones I focused on tuning:
# - `scale_pos_weight`: balances positive/negative class weights — useful given the class imbalance here
# - `subsample`: fraction of rows sampled per boosting round
# - `colsample_bytree` / `colsample_bylevel` / `colsample_bynode`: fraction of columns sampled at the tree/level/node level respectively
# - `max_depth`: maximum tree depth
# - `learning_rate` (`eta`): shrinks each step's weight update for more robust training
# - `gamma`: minimum loss reduction required before a further split is made

# Choose the type of classifier.
xgb_tuned = XGBClassifier(random_state=1,eval_metric='logloss')

# Grid of parameters to choose from
## add from
parameters = {
    "n_estimators": np.arange(10,100,20),
    "scale_pos_weight":[0,1,2,5],
    "subsample":[0.5,0.7,0.9,1],
    "learning_rate":[0.01,0.1,0.2,0.05],
    "gamma":[0,1,3],
    "colsample_bytree":[0.5,0.7,0.9,1],
    "colsample_bylevel":[0.5,0.7,0.9,1]
}

# Type of scoring used to compare parameter combinations
acc_scorer = metrics.make_scorer(metrics.recall_score)

# Run the grid search
grid_obj = GridSearchCV(xgb_tuned, parameters,scoring=acc_scorer,cv=5)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
xgb_tuned = grid_obj.best_estimator_

# Fit the best algorithm to the data.
xgb_tuned.fit(X_train, y_train)

#Using above defined function to get accuracy, recall and precision on train and test set
xgb_tuned_score=get_metrics_score(xgb_tuned)

make_confusion_matrix(xgb_tuned,y_test)

# **My takeaways on the tuned XGBoost model:**
# - Test accuracy dropped slightly compared to the default-parameter version, but recall improved significantly — the model now catches most actual defaulters
# - That recall gain came at the cost of more false positives, which is the right trade-off for this problem
# - Unlike the previous two models, this one isn't overfitting and generalizes well

importances = xgb_tuned.feature_importances_
indices = np.argsort(importances)
feature_names = list(X.columns)

plt.figure(figsize=(12,12))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='violet', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# Checking account balance was the top feature for XGBoost, unlike AdaBoost and Gradient Boosting, which both ranked loan amount highest.

# ## Comparing all models

# defining list of models
models = [abc, abc_tuned, gbc, gbc_init, gbc_tuned, xgb, xgb_tuned]

# defining empty lists to add train and test results
acc_train = []
acc_test = []
recall_train = []
recall_test = []
precision_train = []
precision_test = []

# looping through all the models to get the accuracy, precall and precision scores
for model in models:
    j = get_metrics_score(model,False)
    acc_train.append(np.round(j[0],2))
    acc_test.append(np.round(j[1],2))
    recall_train.append(np.round(j[2],2))
    recall_test.append(np.round(j[3],2))
    precision_train.append(np.round(j[4],2))
    precision_test.append(np.round(j[5],2))

comparison_frame = pd.DataFrame({'Model':['AdaBoost with default paramters','AdaBoost Tuned',
                                          'Gradient Boosting with default parameters','Gradient Boosting with init=AdaBoost',
                                          'Gradient Boosting Tuned','XGBoost with default parameters','XGBoost Tuned'],
                                          'Train_Accuracy': acc_train,'Test_Accuracy': acc_test,
                                          'Train_Recall':recall_train,'Test_Recall':recall_test,
                                          'Train_Precision':precision_train,'Test_Precision':precision_test})
comparison_frame

# ### Business takeaway
# 
# A cost function quantifies the gap between predicted and actual outcomes as a single number a business can optimize against. For a bank, that cost function should reflect the real trade-off between two kinds of mistakes: missing a defaulter (false negative) versus turning down a good applicant (false positive) — and that trade-off shifts with interest rates, regulation, and market conditions.
# 
# I optimized for recall throughout, on the assumption that the cost of a missed defaulter outweighs the cost of a false alarm. But that assumption has a limit: if a model over-corrects and misclassifies too many *good* applicants as defaulters, the bank can lose money even while catching every actual defaulter.
# 
# **Worked example:** say a bank earns 4% interest on loans to non-defaulters and loses 70% of the loan amount on defaults. A model that catches all 10 out of 10 actual defaulters, but misclassifies 180 of 190 good applicants as defaulters, would save $0.70 \times 10 = 7$ (in loan-amount units) from stopped defaults, but lose $0.04 \times 180 = 7.2$ in forgone interest from good applicants turned away — a net loss, despite perfect recall on defaulters. Recall matters, but it isn't the whole picture.
