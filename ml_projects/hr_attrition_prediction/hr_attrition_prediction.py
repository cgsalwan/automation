"""
HR Employee Attrition Prediction — Ensemble Methods

Predicts employee attrition risk using Decision Trees, Bagging, and
Random Forest (default, weighted, and hyperparameter-tuned variants),
optimized for recall since missing an at-risk employee is costlier
than a false positive.
"""

# # HR Employee Attrition Prediction — Ensemble Methods
# 
# *Predicting employee attrition risk using Decision Trees, Bagging, and Random Forest, tuned to prioritize recall for the business use case.*

# <center><img src="https://images.pexels.com/photos/3184360/pexels-photo-3184360.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2" width="800" height="500"></center>
# 
# <b><h2><center>HR Employee Attrition Case Study</center></h2></b>

# ## Problem Statement

# ### Background
# 
# Retaining employees is expensive, and companies often spend heavily on retention initiatives across their entire workforce rather than targeting the employees actually at risk of leaving. This project analyzes HR data to identify the factors that drive attrition, with the goal of predicting which employees are at risk so that retention efforts (and budget) can be targeted more effectively.
# 
# ### Objective
# 
# Analyze employee data to identify the key drivers of attrition, and build a classification model that predicts attrition risk — optimizing for recall, since failing to flag an employee who actually leaves is costlier to the business than a false alarm.
# 
# ### Dataset
# 
# The data contains demographic details, work-related metrics, and an attrition flag.
# 
# * EmployeeNumber - Employee Identifier
# * Attrition - Did the employee attrite?
# * Age - Age of the employee
# * BusinessTravel - Travel commitments for the job
# * DailyRate - Data description not available**
# * Department - Employee Department
# * DistanceFromHome - Distance from work to home (in km)
# * Education - 1-Below College, 2-College, 3-Bachelor, 4-Master,5-Doctor
# * EducationField - Field of Education
# * EmployeeCount - Employee Count in a row
# * EnvironmentSatisfaction - 1-Low, 2-Medium, 3-High, 4-Very High
# * Gender - Employee's gender
# * HourlyRate - Data description not available**
# * JobInvolvement - 1-Low, 2-Medium, 3-High, 4-Very High
# * JobLevel - Level of job (1 to 5)
# * JobRole - Job Roles
# * JobSatisfaction - 1-Low, 2-Medium, 3-High, 4-Very High
# * MaritalStatus - Marital Status
# * MonthlyIncome - Monthly Salary
# * MonthlyRate - Data description not available**
# * NumCompaniesWorked - Number of companies worked at
# * Over18 - Over 18 years of age?
# * OverTime - Overtime?
# * PercentSalaryHike - The percentage increase in salary last year
# * PerformanceRating - 1-Low, 2-Good, 3-Excellent, 4-Outstanding
# * RelationshipSatisfaction - 1-Low, 2-Medium, 3-High, 4-Very High
# * StandardHours - Standard Hours
# * StockOptionLevel - Stock Option Level
# * TotalWorkingYears - Total years worked
# * TrainingTimesLastYear - Number of training attended last year
# * WorkLifeBalance - 1-Low, 2-Good, 3-Excellent, 4-Outstanding
# * YearsAtCompany - Years at Company
# * YearsInCurrentRole - Years in the current role
# * YearsSinceLastPromotion - Years since the last promotion
# * YearsWithCurrManager - Years with the current manager
# 
# ** **In the real world, you will not find definitions for some of your variables. It is a part of the analysis to figure out what they might mean.**

# ## Importing necessary libraries

# Libraries to help with reading and manipulating data
import numpy as np
import pandas as pd

# libaries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Library to split data
from sklearn.model_selection import train_test_split

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score, roc_auc_score
import scipy.stats as stats

import warnings
warnings.filterwarnings('ignore')

# ## Reading the dataset

hr=pd.read_csv("HR_Employee_Attrition-1.csv")

# copying data to another varaible to avoid any changes to original data
data=hr.copy()

# ## Overview of the dataset

# ### View the first and last 5 rows of the dataset.

data.head()

data.tail()

# ### Understand the shape of the dataset.

data.shape

# * The dataset has 2940 rows and 35 columns of data

# ### Check the data types of the columns for the dataset.

data.info()

# **Observations -**
# * There are no null values in the dataset.
# * We can convert the object type columns to categories.
# 
# `converting "objects" to "category" reduces the data space required to store the dataframe`

# ### Fixing the data types

cols = data.select_dtypes(['object'])
cols.columns

for i in cols.columns:
    data[i] = data[i].astype('category')

data.info()

# `we can see that the memory usage has decreased from 804 KB to 624.4 KB`, this technique is generally useful for bigger datasets.

# ### Summary of the dataset

data.describe().T

# * EmployeeNumber is an ID variable and not useful for predictive modelling.
# * Age of the employees range from 18 to 60 years and the average age is 36 years.
# * EmployeeCount has only 1 as the value in all the rows and can be dropped as it will not be adding any information to our analysis.
# * Standard Hours has only 80 as the value in all the rows and can be dropped as it will not be adding any information to our analysis.
# * Hourly rate has a huge range, but we do not know what this variable stands for, yet. The same goes for daily and monthly rates.
# * Monthly Income has a high range and the difference in mean and median indicate the presence of outliers.

data.describe(include=['category']).T

# * Attrition is our target variable with 84% records 'No' or employee will not attrite.
# * Majority of the employees have low business travel requirements
# * Majority of the employees are from the Research and Development department.
# * All employees are over 18 years of age -  we can drop this variable as it will not be adding any information to our analysis.
# * There are more male employees than female employees.

# **Dropping columns which are not adding any information.**

data.drop(['EmployeeNumber','EmployeeCount','StandardHours','Over18'],axis=1,inplace=True)

# **Let's look at the unique values of all the categories**

cols_cat= data.select_dtypes(['category'])

for i in cols_cat.columns:
    print('Unique values in',i, 'are :')
    print(cols_cat[i].value_counts())
    print('*'*50)

df = data.copy()

# ## <a name='link2'>Exploratory Data Analysis (EDA) Summary</a>

# ### **Note**: The EDA section has been covered multiple times in the previous case studies. In this case study, we will mainly focus on the model building aspects. We will only be looking at the key observations from EDA. The detailed EDA can be found in the <a href = #link1>appendix section</a>.

# **The below functions need to be defined to carry out the EDA.**

def histogram_boxplot(data, feature, figsize=(15, 10), kde=False, bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (15,10))
    kde: whether to show the density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )  # boxplot will be created and a triangle will indicate the mean value of the column
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    )  # For histogram
    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram

# function to create labeled barplots


def labeled_barplot(data, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(data[feature])  # length of the column
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 2, 6))
    else:
        plt.figure(figsize=(n + 2, 6))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n],
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage

    plt.show()  # show the plot

# function to plot stacked bar chart


def stacked_barplot(data, predictor, target):
    """
    Print the category counts and plot a stacked bar chart

    data: dataframe
    predictor: independent variable
    target: target variable
    """
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    tab1 = pd.crosstab(data[predictor], data[target], margins=True).sort_values(
        by=sorter, ascending=False
    )
    print(tab1)
    print("-" * 120)
    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(
        by=sorter, ascending=False
    )
    tab.plot(kind="bar", stacked=True, figsize=(count + 1, 5))
    plt.legend(
        loc="lower left",
        frameon=False,
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()

# ### Univariate analysis

# #### Observations on NumCompaniesWorked

histogram_boxplot(data,'NumCompaniesWorked')

# * On average, people have worked at 2.5 companies. Median is 2.
# * Most people have worked at only 1 company.
# * Nearly 350 employees have worked at 0 companies, clearly this means this variable indicates the number of companies worked at before joining ours.
# * There is an outlier employee who has changed 9 companies.

# #### Observations on YearsInCurrentRole

histogram_boxplot(data,'YearsSinceLastPromotion')

# * There are a few outliers in this right-skewed distribution, these are probably the people at the highest positions.
# * Most employees have had a promotion in the last 2 years.
# * 0 years since last promotion indicates many employees were recently promoted.

# ### Observations on JobRole

labeled_barplot(data, "JobRole", perc=True)

# * 22.2% of employees are Sales Executives followed by 20% of Research Scientists.

# #### Observations on Attrition

labeled_barplot(data, "Attrition", perc=True)

# * 16% of the data points represent the employees who are going to attrite.

# ### Bivariate Analysis

# #### Correlation check

plt.figure(figsize=(20,10))
sns.heatmap(data.corr(numeric_only=True),annot=True,vmin=-1,vmax=1,fmt='.2f',cmap="Spectral")
plt.show()

# * There are a few variables that are correlated with each other but there are no surprises here.
# * Unsurprisingly, TotalWorkingYears is highly correlated to Job Level (i.e., the longer you work the higher job level you achieve).
# * HourlyRate, DailyRate, and MonthlyRate are completely uncorrelated with each other which makes it harder to understand what these variables might represent.
# * MonthlyIncome is highly correlated to Job Level.
# * Age is positively correlated JobLevel and Education (i.e., the older an employee is, the more educated and at a higher job level they are).
# * Work-life Balance is correlated with none of the numeric values.

# #### Attrition vs Earnings of employee

cols = data[['DailyRate','HourlyRate','MonthlyRate','MonthlyIncome','PercentSalaryHike']].columns.tolist()
plt.figure(figsize=(10,10))

for i, variable in enumerate(cols):
                     plt.subplot(3,2,i+1)
                     sns.boxplot(x=data["Attrition"],y=data[variable],palette="PuBu")
                     plt.tight_layout()
                     plt.title(variable)
plt.show()

# * Employees having lower Daily rate and less monthly wage are more likely to attrite.
# * Monthly rate and the hourly rate doesn't seem to have any effect on attrition.
# * Lesser salary hike also contributes to attrition.

# #### Attrition vs Previous job roles

cols = data[['NumCompaniesWorked','TotalWorkingYears']].columns.tolist()
plt.figure(figsize=(10,10))

for i, variable in enumerate(cols):
                     plt.subplot(3,2,i+1)
                     sns.boxplot(x=data["Attrition"],y=data[variable],palette="PuBu")
                     plt.tight_layout()
                     plt.title(variable)
plt.show()

# * Employees who have worked in more companies generally tend to switch more jobs hence attriting.
# * Employees who attrite generally have lesser years of experience.

stacked_barplot(data, "BusinessTravel", "Attrition")

# * As the travel frequency increases, the Attrition rate increases.
# * There's ~22% probability of employees attriting who travel frequently.

stacked_barplot(data,"EnvironmentSatisfaction","Attrition")

# * Employees who say they have low satisfaction with their work environments are likely to attrite.
# * There's a ~40% probability of attrition among employees with low ratings for environment satisfaction.

stacked_barplot(data,"JobInvolvement","Attrition")

# * Job Involvement looks like a very strong indicator of attrition.
# * Higher the job involvement, greater is the chance that the employee will stay with us and not attrite.
# * Employees unhappy with their job involment have ~55% probability of attriting (those who rated 0 and 1).
# * Further investigation to understand how this variable was collected will give more insights.

stacked_barplot(data,"JobRole","Attrition")

# * Sales Executives have an attrition probability of >40%.
# * Laboratory Technicians and Human Resource personnel also have high probabilities of attrition.
# * Attrition probability among Research Directors, Manufacturing directors Healthcare representatives, and Managers is much lower than the average attrition probability of 16%.

stacked_barplot(data,"JobSatisfaction","Attrition")

# * As Job satisfaction increases, attrition probability decreases. This is intuitive but the attrition probability of people who rate 2 and 3 being almost the same is peculiar.  

stacked_barplot(data,"OverTime","Attrition")

# * Employees who work overtime tend to attrite more.
# * There a ~35% probability of attrition among employees working overtime.

stacked_barplot(data,"StockOptionLevel","Attrition")

# * ~22% Employees with highest and lowest stock options attrite the more than others.
# * Company should investigate more on why employees with highest stock options are attriting and take this as an opportunity to re-consider their stocks policy.

stacked_barplot(data,"WorkLifeBalance","Attrition")

# * Low work-life balance rating leads people to attrite, this is a good factor to preempt at attrition risk employees.

# **Checking if performace rating and salary hike are related-**

sns.boxplot(x=data['PerformanceRating'],y=data['PercentSalaryHike'])
plt.show()

plt.figure(figsize=(15,5))
sns.boxplot(x=data['PerformanceRating'],y=data['PercentSalaryHike'],hue=data['JobRole'])
plt.show()

# **Observations-**
# * Salary hikes are a function of Performance ratings.
# * We have to investigate why the employees who get Excellent(3) and Outstanding(4) Performance rating attrite and how then can they be retained.

# ## Data Preprocessing

# ### Outlier Detection and Treatment

# outlier detection using boxplot
numeric_columns = data.select_dtypes(include=np.number).columns.tolist()


plt.figure(figsize=(15, 12))

for i, variable in enumerate(numeric_columns):
    plt.subplot(6, 4, i + 1)
    plt.boxplot(data[variable], whis=1.5)
    plt.tight_layout()
    plt.title(variable)

plt.show()

# - There are quite a few outliers in the data.
# - However, we will not treat them as they are proper values.

# ### Data Preparataion for model building

# * When classification problems exhibit a significant imbalance in the distribution of the target classes, it is good to use stratified sampling to ensure that relative class frequencies are approximately preserved in train and test sets.
# * This is done using the `stratify` parameter in the train_test_split function.

data['Attrition'] = data['Attrition'].apply(lambda x : 1 if x=='Yes' else 0)

X = data.drop(['Attrition'],axis=1)
y = data['Attrition']

X = pd.get_dummies(X,drop_first=True)

# Splitting data into training and test set:
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.3, random_state=1,stratify=y)
print(X_train.shape, X_test.shape)

y.value_counts(1)

y_test.value_counts(1)

# **Let's define function to provide metric scores(accuracy,recall and precision) on train and test set and a function to show confusion matrix so that we do not have use the same code repetitively while evaluating models.**

# defining a function to compute different metrics to check performance of a classification model built using sklearn
def model_performance_classification_sklearn(model, predictors, target):
    """
    Function to compute different metrics to check classification model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    """

    # predicting using the independent variables
    pred = model.predict(predictors)

    acc = accuracy_score(target, pred)  # to compute Accuracy
    recall = recall_score(target, pred)  # to compute Recall
    precision = precision_score(target, pred)  # to compute Precision
    f1 = f1_score(target, pred)  # to compute F1-score

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {
            "Accuracy": acc,
            "Recall": recall,
            "Precision": precision,
            "F1": f1,
        },
        index=[0],
    )

    return df_perf

def confusion_matrix_sklearn(model, predictors, target):
    """
    To plot the confusion_matrix with percentages

    model: classifier
    predictors: independent variables
    target: dependent variable
    """
    y_pred = model.predict(predictors)
    cm = confusion_matrix(target, y_pred)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(2, 2)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

# ## Model Building

# ### Model evaluation criterion
# 
# ### Model can make wrong predictions as:
# 1. Predicting an employee will attrite and the employee doesn't attrite
# 2. Predicting an employee will not attrite and the employee attrites
# 
# ### Which case is more important?
# * Predicting that employee will not attrite but he attrites i.e. losing on a valuable employee or asset.
# 
# ### How to reduce this loss i.e need to reduce False Negatives?
# * Company wants Recall to be maximized, greater the Recall higher the chances of minimizing false negatives. Hence, the focus should be on increasing Recall or minimizing the false negatives or in other words identifying the true positives(i.e. Class 1) so that the company can provide incentives to control attrition rate especially for top-performers thereby optimizing the overall project cost in retaining the best talent.

# ### Decision Tree Model

# * We will build our model using the DecisionTreeClassifier function. Using default 'gini' criteria to split.
# * If the frequency of class A is 10% and the frequency of class B is 90%, then class B will become the dominant class and the decision tree will become biased toward the dominant classes.
# 
# * In this case, we can pass a dictionary {0:0.17,1:0.83} to the model to specify the weight of each class and the decision tree will give more weightage to class 1.
# 
# * class_weight is a hyperparameter for the decision tree classifier.

dtree = DecisionTreeClassifier(criterion='gini',class_weight={0:0.17,1:0.83},random_state=1)

dtree.fit(X_train, y_train)

confusion_matrix_sklearn(dtree, X_train, y_train)

# **Confusion Matrix -**
# 
# * Employee left and the model predicted it correctly that is employee will attrite :  True Positive (observed=1,predicted=1)
# 
# * Employee didn't leave and the model predicted employee will attrite : False Positive (observed=0,predicted=1)
# 
# * Employee didn't leave and the model predicted employee will not attrite : True Negative (observed=0,predicted=0)
# 
# * Employee left and the model predicted that employee won't : False Negative (observed=1,predicted=0)

dtree_model_train_perf=model_performance_classification_sklearn(dtree, X_train, y_train)
print("Training performance \n",dtree_model_train_perf)

confusion_matrix_sklearn(dtree, X_test, y_test)

dtree_model_test_perf=model_performance_classification_sklearn(dtree, X_test, y_test)
print("Testing performance \n",dtree_model_test_perf)

# * Decision tree is working well on the training data but is not able to generalize well on the test data concerning the recall.

# ### Bagging Classifier

bagging = BaggingClassifier(random_state=1)
bagging.fit(X_train,y_train)

confusion_matrix_sklearn(bagging, X_train, y_train)

bagging_model_train_perf=model_performance_classification_sklearn(bagging, X_train, y_train)
print("Training performance \n",bagging_model_train_perf)

confusion_matrix_sklearn(bagging, X_test, y_test)

bagging_model_test_perf=model_performance_classification_sklearn(bagging, X_test, y_test)
print("Testing performance \n",bagging_model_test_perf)

# * Bagging classifier is overfitting on the training set and is performing poorly on the test set in terms of recall.

# **Bagging Classifier with weighted decision tree**

bagging_wt = BaggingClassifier(DecisionTreeClassifier(criterion='gini',class_weight={0:0.17,1:0.83},random_state=1),random_state=1)
bagging_wt.fit(X_train,y_train)

confusion_matrix_sklearn(bagging_wt,X_train,y_train)

bagging_wt_model_train_perf=model_performance_classification_sklearn(bagging_wt,X_train,y_train)
print("Training performance \n",bagging_wt_model_train_perf)

confusion_matrix_sklearn(bagging_wt,X_test,y_test)

bagging_wt_model_test_perf=model_performance_classification_sklearn(bagging_wt, X_test, y_test)
print("Testing performance \n",bagging_wt_model_test_perf)

# * Bagging classifier with a weighted decision tree is giving very good accuracy and prediction but is not able to generalize well on test data in terms of recall.

# ### Random Forest

rf = RandomForestClassifier(random_state=1)
rf.fit(X_train,y_train)

confusion_matrix_sklearn(rf,X_train,y_train)

rf_model_train_perf=model_performance_classification_sklearn(rf,X_train,y_train)
print("Training performance \n",rf_model_train_perf)

confusion_matrix_sklearn(rf,X_test,y_test)

rf_model_test_perf=model_performance_classification_sklearn(rf,X_test,y_test)
print("Testing performance \n",rf_model_test_perf)

# * Random Forest has performed well in terms of accuracy and precision, but it is not able to generalize well on the test data in terms of recall.

# **Random forest with class weights**

rf_wt = RandomForestClassifier(class_weight={0:0.17,1:0.83}, random_state=1)
rf_wt.fit(X_train,y_train)

confusion_matrix_sklearn(rf_wt, X_train,y_train)

rf_wt_model_train_perf=model_performance_classification_sklearn(rf_wt, X_train,y_train)
print("Training performance \n",rf_wt_model_train_perf)

confusion_matrix_sklearn(rf_wt, X_test,y_test)

rf_wt_model_test_perf=model_performance_classification_sklearn(rf_wt, X_test,y_test)
print("Testing performance \n",rf_wt_model_test_perf)

# * There is not much improvement in metrics of weighted random forest as compared to the unweighted random forest.

# ## Tuning Models

# ### Using GridSearch for Hyperparameter tuning model

# * Hyperparameter tuning is also tricky in the sense that there is no direct way to calculate how a change in the
#   hyperparameter value will reduce the loss of your model, so we usually resort to experimentation. i.e we'll use Grid search
# * Grid search is a tuning technique that attempts to compute the optimum values of hyperparameters.
# * It is an exhaustive search that is performed on a the specific parameter values of a model.
# * The parameters of the estimator/model used to apply these methods are optimized by cross-validated grid-search over a parameter grid.

# ### Tuning Decision Tree

# Choose the type of classifier.
dtree_estimator = DecisionTreeClassifier(class_weight={0:0.17,1:0.83},random_state=1)

# Grid of parameters to choose from
parameters = {'max_depth': np.arange(2,30),
              'min_samples_leaf': [1, 2, 5, 7, 10],
              'max_leaf_nodes' : [2, 3, 5, 10,15],
              'min_impurity_decrease': [0.0001,0.001,0.01,0.1]
             }

# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.recall_score)

# Run the grid search
grid_obj = GridSearchCV(dtree_estimator, parameters, scoring=scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
dtree_estimator = grid_obj.best_estimator_

# Fit the best algorithm to the data.
dtree_estimator.fit(X_train, y_train)

confusion_matrix_sklearn(dtree_estimator, X_train,y_train)

dtree_estimator_model_train_perf=model_performance_classification_sklearn(dtree_estimator, X_train,y_train)
print("Training performance \n",dtree_estimator_model_train_perf)

confusion_matrix_sklearn(dtree_estimator, X_test,y_test)

dtree_estimator_model_test_perf=model_performance_classification_sklearn(dtree_estimator, X_test, y_test)
print("Testing performance \n",dtree_estimator_model_test_perf)

# * Overfitting in decision tree has reduced but the recall has also reduced.

# ### Tuning Bagging Classifier

# grid search for bagging classifier
cl1 = DecisionTreeClassifier(class_weight={0:0.13,1:0.87},random_state=1)
param_grid = {
              'n_estimators':[5,7,15,51,101],
              'max_features': [0.7,0.8,0.9,1]
             }

grid = GridSearchCV(BaggingClassifier(cl1, random_state=1,bootstrap=True), param_grid=param_grid, scoring = 'recall', cv = 5)
grid.fit(X_train, y_train)

## getting the best estimator
bagging_estimator  = grid.best_estimator_
bagging_estimator.fit(X_train,y_train)

confusion_matrix_sklearn(bagging_estimator, X_train,y_train)

bagging_estimator_model_train_perf=model_performance_classification_sklearn(bagging_estimator, X_train,y_train)
print("Training performance \n",bagging_estimator_model_train_perf)

confusion_matrix_sklearn(bagging_estimator, X_test,y_test)

bagging_estimator_model_test_perf=model_performance_classification_sklearn(bagging_estimator, X_test, y_test)
print("Testing performance \n",bagging_estimator_model_test_perf)

# * Recall has improved but the accuracy and precision of the model has dropped drastically which is an indication that overall the model is making many mistakes.

# ### Tuning Random Forest

# Choose the type of classifier.
rf_estimator = RandomForestClassifier(random_state=1)

# Grid of parameters to choose from
parameters = {
        "n_estimators": [110,251,501],
        "min_samples_leaf": np.arange(1, 6,1),
        "max_features": [0.7,0.9,'log2','auto'],
        "max_samples": [0.7,0.9,None],
}

# Run the grid search
grid_obj = GridSearchCV(rf_estimator, parameters, scoring='recall',cv=5)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
rf_estimator = grid_obj.best_estimator_

# Fit the best algorithm to the data.
rf_estimator.fit(X_train, y_train)

confusion_matrix_sklearn(rf_estimator, X_train,y_train)

rf_estimator_model_train_perf=model_performance_classification_sklearn(rf_estimator, X_train,y_train)
print("Training performance \n",rf_estimator_model_train_perf)

confusion_matrix_sklearn(rf_estimator, X_test,y_test)

rf_estimator_model_test_perf=model_performance_classification_sklearn(rf_estimator, X_test, y_test)
print("Testing performance \n",rf_estimator_model_test_perf)

# * Random forest after tuning has given same performance as un-tuned random forest.

# ### Comparing all the models

# training performance comparison
bagging_wt_model_train_perf=model_performance_classification_sklearn(bagging_wt,X_train,y_train)
print("Training performance \n",bagging_wt_model_train_perf)
models_train_comp_df = pd.concat(
    [dtree_model_train_perf.T,bagging_model_train_perf.T, bagging_wt_model_train_perf.T,rf_model_train_perf.T,
    rf_wt_model_train_perf.T,dtree_estimator_model_train_perf.T, bagging_estimator_model_train_perf.T,
     rf_estimator_model_train_perf.T],
    axis=1,
)
models_train_comp_df.columns = [
    "Decision Tree",
    "Bagging Classifier",
    "Weighted Bagging Classifier",
    "Random Forest Classifier",
    "Weighted Random Forest Classifier",
    "Decision Tree Estimator",
    "Bagging Estimator",
    "Random Forest Estimator"]
print("Training performance comparison:")
models_train_comp_df

# training performance comparison

models_test_comp_df = pd.concat(
    [dtree_model_test_perf.T,bagging_model_test_perf.T, bagging_wt_model_test_perf.T,rf_model_test_perf.T,
    rf_wt_model_test_perf.T,dtree_estimator_model_test_perf.T, bagging_estimator_model_test_perf.T,
     rf_estimator_model_test_perf.T],
    axis=1,
)
models_test_comp_df.columns = [
    "Decision Tree",
    "Bagging Classifier",
    "Weighted Bagging Classifier",
    "Random Forest Classifier",
    "Weighted Random Forest Classifier",
    "Decision Tree Estimator",
    "Bagging Estimator",
    "Random Forest Estimator"]
print("Testing performance comparison:")
models_test_comp_df

# * Decision tree performed well on training and test set.
# * Bagging classifier overfitted the data before and after tuning.
# * Random Forest with default parameters performed same as after tuning - As the final results depend on the parameters used/checked using GridSearchCV, There may be yet better parameters which may result in a better performance.

# ### Feature importance of Random Forest

# importance of features in the tree building ( The importance of a feature is computed as the
#(normalized) total reduction of the criterion brought by that feature. It is also known as the Gini importance )

print (pd.DataFrame(rf.feature_importances_, columns = ["Imp"], index = X_train.columns).sort_values(by = 'Imp', ascending = False))

feature_names = X_train.columns

importances = rf_estimator.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12,12))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='violet', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# * Monthly income is the most important feature for prediction followed by Overtime, Daily Rate and Age.

# ## Business Insights and Recommendations

# 
# * We have been able to build a predictive model:
#   a) that company can deploy this model to identify employees who are at the risk of attrition.
#   b) that company can use to find the drivers of attrition.
#   c) based on which company can take appropriate actions to build better retention policies.
# 
# * Factors that drive attrition - Monthly Income, Overtime, and Age.
# * Monthly Income: Employees with lower income attrite more, which is also logical as they might get offers with higher pay in different organizations - the company should make sure that all employees are compensated based on industry standards.
# 
# * Overtime: Those employees who have to work overtime are the ones who attrite more - the company can provide some additional incentives to such employees to retain them.
# 
# * Age: Younger employees are the ones that attrite more- the company can make sure the new joiners have a friendly environment and better opportunities for excelling in their career.
# 
# * Distance From home is also an important factor for attrition - employees traveling more distance to reach the workplace are the ones attriting. For such employees, the company can provide cab facilities so that the commute of employees gets easier.
# 
# * As work-related travel frequency increases, Attrition rate also increases - the company should
# 
# * Training doesn't seem to have an impact on attrition- the company needs to investigate more here, if training does not impact employee retention then better cost planning can be done.
# 
# * Employee with more experience and the employees working for most years in the company is the loyal one's and generally do not attrite.
# 
# * Highest attrition is in the Sales department more research should go into this to check what is wrong in the sales department?
# 
# * Our data collection technique is working well as the ratings given by employees in -Environment Satisfaction, Job Satisfaction, Relationship Satisfaction, and Work-Life Balance shows a difference significant difference between attriting and non-attriting employees. These scales can act as a preliminary step to understand the dissatisfaction of employees - Lower the rating higher are the chances of attrition.

# ## <a name='link1'>Appendix: Detailed Exploratory Data Analysis (EDA)</a>

# ### Univariate analysis

# ### Observations on Age

histogram_boxplot(df,'Age')

# * Age is looking normally distributed, with a hint of right skew.

# ### Observations on DailyRate

histogram_boxplot(df,'DailyRate')

# * The daily rate has a fairly uniform distribution with a mean and median at 800.

# ### Observations on DistanceFromHome

histogram_boxplot(df,'DistanceFromHome')

# * This is a right-skewed distribution.
# * Most people live close to work but we do have some employees that live at a distance.

# ### Observations on HourlyRate

histogram_boxplot(df,'HourlyRate')

# * Quite inline with the daily rate, the hourly rate is also fairly uniformly distributed with mean and median at 65.

# ### Observations on MonthlyIncome

histogram_boxplot(df,'MonthlyIncome')

# * Income distribution across a unit is almost always right-skewed and similar is the case here.
# * As expected, from the boxplot we can see that there are a few outliers who earn much higher than the rest of the employees.

# ### Observations on MonthlyRate

histogram_boxplot(df,'MonthlyRate')

# * Monthly rate also has a uniform distribution with mean and median close to 14500.

# ### Observations on NumCompaniesWorked

histogram_boxplot(df,'NumCompaniesWorked')

# * On average, people have worked at 2.5 companies. Median is 2.
# * Most people have worked at only 1 company.
# * Nearly 350 employees have worked at 0 companies, clearly this means this variable indicates the number of companies worked at before joining ours.
# * There is an outlier employee who has changed 9 companies.

# ### Observations on PercentSalaryHike

histogram_boxplot(df,'PercentSalaryHike')

# * Salary hike is Right skewed. We shall check later if the hike percentage is a function of performance rating or job level.
# * Mean percentage salary hike is 15%.

# ### Observations on TotalWorkingYears

histogram_boxplot(df,'TotalWorkingYears')

# * Age of the employees had a hint of right skew but work experience has a significant right skew.
# * From the boxplot, we can observe that this variable contains a few outliers also.

# ### Observations on YearsAtCompany

histogram_boxplot(df,'YearsAtCompany')

# * Significant right skew in the data.
# * The outliers are probably the most loyal employees and it suggests that they would be less likely to attrite.

# ### Observations on YearsInCurrentRole

histogram_boxplot(df,'YearsInCurrentRole')

# * This distribution has three peaks at 0, 2, and 7.
# * There are a few outliers that have stayed in the same role for more than 15 years.

# ### Observations on YearsInCurrentRole

histogram_boxplot(df,'YearsSinceLastPromotion')

# * There are a few outliers in this right-skewed distribution, these are probably the people at the highest positions.
# * Most employees have had a promotion in the last 2 years.
# * 0 years since last promotion indicates many employees were recently promoted.

# ### Observations on YearsWithCurrManager

histogram_boxplot(df,'YearsWithCurrManager')

# * This distribution is very similar to that of 'Years in current role' and that makes sense.
# * There are a few outliers in this variable.

# ### Observations on BusinessTravel

labeled_barplot(df, "BusinessTravel", perc=True)

# * 71% of the employees have travel rarely and 18.8% employees have to travel frequently.

# ### Observations on Department

labeled_barplot(df, "Department", perc=True)

# * 65.4% of employees in data are from R&D department followed by 30.3% in sales.

# ### Observations on EducationField

labeled_barplot(df, "EducationField", perc=True)

# * 41.2% of the employees are from a Life Sciences background followed by 31.6% from a medical background.

# ### Observations on Gender

labeled_barplot(df, "Gender", perc=True)

# * 60% of male employees and 40% of female employees.

# ### Observations on JobRole

labeled_barplot(df, "JobRole", perc=True)

# * 22.2% of employees are Sales Executives followed by 20% of Research Scientists.

# ### Observations on MaritalStatus

labeled_barplot(df, "MaritalStatus", perc=True)

# * 45.8% of the employees are married followed by 32% singles.

# ### Observations on OverTime

labeled_barplot(df, "OverTime", perc=True)

# * 71.7% of the employees are not ready to work over time.

# ### Observations on Attrition

labeled_barplot(df, "Attrition", perc=True)

# * 16% of the data points represent the employees who are going to attrite.

# ### Bivariate Analysis

plt.figure(figsize=(20,10))
sns.heatmap(df.corr(),annot=True,vmin=-1,vmax=1,fmt='.2f',cmap="Spectral")
plt.show()

# * There are a few variables that are correlated with each other but there are no surprises here.
# * Unsurprisingly, TotalWorkingYears is highly correlated to Job Level (i.e., the longer you work the higher job level you achieve).
# * HourlyRate, DailyRate, and MonthlyRate are completely uncorrelated with each other which makes it harder to understand what these variables might represent.
# * MonthlyIncome is highly correlated to Job Level.
# * Age is positively correlated JobLevel and Education (i.e., the older an employee is, the more educated and at a higher job level they are).
# * Work-life Balance is correlated with none of the numeric values.

sns.pairplot(df,hue='Attrition')
plt.show()

# * We can see varying distributions in variables for Attrition, we should investigate it further.

# ### Attrition vs Earnings of employee

cols = df[['DailyRate','HourlyRate','MonthlyRate','MonthlyIncome','PercentSalaryHike']].columns.tolist()
plt.figure(figsize=(10,10))

for i, variable in enumerate(cols):
                     plt.subplot(3,2,i+1)
                     sns.boxplot(x=df["Attrition"],y=df[variable],palette="PuBu")
                     plt.tight_layout()
                     plt.title(variable)
plt.show()

# * Employees having lower Daily rate and less monthly wage are more likely to attrite.
# * Monthly rate and the hourly rate doesn't seem to have any effect on attrition.
# * Lesser salary hike also contributes to attrition.

# ### Attrition vs Years working in company

cols = df[['YearsAtCompany', 'YearsInCurrentRole',
       'YearsSinceLastPromotion', 'YearsWithCurrManager','TrainingTimesLastYear']].columns.tolist()
plt.figure(figsize=(10,10))

for i, variable in enumerate(cols):
                     plt.subplot(3,2,i+1)
                     sns.boxplot(x=df["Attrition"],y=df[variable],palette="PuBu")
                     plt.tight_layout()
                     plt.title(variable)
plt.show()

# * Those employees who have spent less time at a company, in a current role or with a manager have higher chances of attriting.
# * Training doesn't seem to have an impact on Attrition.

# ### Attrition vs Previous job roles

cols = df[['NumCompaniesWorked','TotalWorkingYears']].columns.tolist()
plt.figure(figsize=(10,10))

for i, variable in enumerate(cols):
                     plt.subplot(3,2,i+1)
                     sns.boxplot(x=df["Attrition"],y=df[variable],palette="PuBu")
                     plt.tight_layout()
                     plt.title(variable)
plt.show()

# * Employees who have worked in more companies generally tend to switch more jobs hence attriting.
# * Employees who attrite generally have lesser years of experience.

cols = df[['Age','DistanceFromHome','Education']].columns.tolist()
plt.figure(figsize=(10,10))

for i, variable in enumerate(cols):
                     plt.subplot(3,2,i+1)
                     sns.boxplot(df["Attrition"],df[variable],palette="PuBu")
                     plt.tight_layout()
                     plt.title(variable)
plt.show()

# * Employees who have to travel a more distance from their home attrite more.
# * There is no difference in age and education of attriting and non-attriting employees.

stacked_barplot(df, "BusinessTravel", "Attrition")

# * As the travel frequency increases, the Attrition rate increases.
# * There's ~22% probability of employees attriting who travel frequently.

stacked_barplot(df, "Department", "Attrition")

# * We saw earlier that majority of the employees work for the R&D department. The probability of attrition is least there.
# * Both Sales and HR have similar Attrition probability.

stacked_barplot(df,"EducationField","Attrition")

# * 25% of employees educated in human resources attrite.
# * Attrition probability is also high in the case of marketing and technical degree holders.

stacked_barplot(df,"EnvironmentSatisfaction","Attrition")

# * Employees who say they have low satisfaction with their work environments are likely to attrite.
# * There's a ~40% probability of attrition among employees with low ratings for environment satisfaction.

stacked_barplot(df,"JobInvolvement","Attrition")

# * Job Involvement looks like a very strong indicator of attrition.
# * Higher the job involvement, greater is the chance that the employee will stay with us and not attrite.
# * Employees unhappy with their job involment have ~55% probability of attriting (those who rated 0 and 1).
# * Further investigation to understand how this variable was collected will give more insights.

stacked_barplot(df,"JobLevel","Attrition")

# * The trend is not very clear here, but it is visible that people at lower job levels are more likely to attrite.

stacked_barplot(df,"JobRole","Attrition")

# * Sales Executives have an attrition probability of >40%.
# * Laboratory Technicians and Human Resource personnel also have high probabilities of attrition.
# * Attrition probability among Research Directors, Manufacturing directors Healthcare representatives, and Managers is much lower than the average attrition probability of 16%.

stacked_barplot(df,"JobSatisfaction","Attrition")

# * As Job satisfaction increases, attrition probability decreases. This is intuitive but the attrition probability of people who rate 2 and 3 being almost the same is peculiar.  

stacked_barplot(df,"MaritalStatus","Attrition")

# * Singles attrite more than married and divorced employees.
# * One of the reasons here can be that single employees are younger and tend to explore different jobs.

stacked_barplot(df,"OverTime","Attrition")

# * Employees who work overtime tend to attrite more.
# * There a ~35% probability of attrition among employees working overtime.

stacked_barplot(df,"RelationshipSatisfaction","Attrition")

# * Low relationship satisfaction rating does indicate more probability of attrition, but we need to investigate further which relationships do this variable indicate.

stacked_barplot(df,"StockOptionLevel","Attrition")

# * ~22% Employees with highest and lowest stock options attrite the more than others.
# * Company should investigate more on why employees with highest stock options are attriting and take this as an opportunity to re-consider their stocks policy.

stacked_barplot(df,"WorkLifeBalance","Attrition")

# * Low work-life balance rating leads people to attrite, this is a good factor to preempt at attrition risk employees.

# **Checking if performace rating and salary hike are related-**

sns.boxplot(df['PerformanceRating'],df['PercentSalaryHike'])
plt.show()

plt.figure(figsize=(15,5))
sns.boxplot(df['PerformanceRating'],df['PercentSalaryHike'],hue=df['JobRole'])
plt.show()

# **Observations-**
# * Salary hikes are a function of Performance ratings.
# * We have to investigate why the employees who get Excellent(3) and Outstanding(4) Performance rating attrite and how then can they be retained.

# ### <a id='link1'>Summary of EDA</a>
# **Data Description:**
# 
# * The dataset has 2940 rows and 35 columns of data.
# * There are no null values in the dataset.
# * Attrition, BusinessTravel, Department, EducationField, Gender, JobRole, MaritalStatus, Over18, OverTime are of object data type while others are of integer data type.
# 
# **Data Cleaning:**
# 
# * EmployeeNumber is an ID variable and not useful for predictive modeling.
# * EmployeeCount has only 1 as the value in all the rows and can be dropped as it will not be adding any information to our analysis.
# * Standard Hours has only 80 as the value in all the rows and can be dropped as it will not be adding any information to our analysis.
# * As all the employees are aged more than 18, hence all the values in the Over18 column are 1. This variable was dropped.
# 
# **Observations from EDA:**
# 
# * `age`: The variable Age is normally distributed with an average equal to 37 years. Age is positively correlated JobLevel and Education (i.e., the older an employee is, the more educated and at a higher job level they are).
# * `DailyRate`: It is having a fairly uniform distribution with an average of 800. Employees having lower Daily rates and less monthly wage are more likely to attrite.
# * `DistanceFromHome`: This shows a right-skewed distribution. Most people stay at a close distance from home with few also leaving away from work. Employees who have to travel a more distance from their home attrite more.
# * `HourlyRate`: It is an almost uniformly distributed variable. It has mean and median approx equal to 65. The hourly rate doesn't have much impact on attrition.
# * `MonthlyIncome`: Monthly Income distribution is almost right-skewed. Few employees are earning much higher than the rest of the employees.  MonthlyIncome is highly correlated to Job Level (0.95).
# * `MonthlyRate`: It is a uniformly distributed variable with a median close to 14500. The monthly rate doesn't have much impact on attrition.
# * `NumCompaniesWorked`: On average people have worked at 2.5 companies with a median of 2 companies. The majority of people have worked in only one company. Approx 350 people are freshers. There are employees worked at 9 companies. They will be the outliers.
# * `PercentSalaryHike`:  It has a right-skewed distribution. It is correlated with performance rating with a coefficient of 0.77. Lesser salary hike also contributes to attrition.
# * `TotalWorkingYears`: Work experience is having a significant right skew. It also contains few outliers.
# * `YearsAtCompany`: Significantly skewed towards the right. Also contains outliers.
# * `YearsInCurrentRole`: It is having few outliers. The lower whisker coincides with the first quartile.
# * `YearsWithCurrManager`: It is a right-skewed variable with few outliers too.
# * `BusinessTravel`: Almost 71% of employees have travel rarely and 19% travel frequently. As the travel frequency increases of an employee increases, the Attrition rate increases. There's a ~22% probability of employees attriting who travel frequently.
# * `Department`: The R&D department consists of almost 65% of employees.
# * `EducationField`:People with Life science background is dominant over others with almost 41% count.
# * `Gender`: 60% of people are male while the rest are female.
# * `JobRole`: Almost 22% of people are sales executives followed by 20% research scientists.
# * `MaritalStatus`: Almost 46% of people are married and 32% are single.
# * `OverTime`: Only almost 29% of people are ready to do overtime. Employees who work overtime tend to attrite more. There a ~35% probability of attrition among employees working overtime.
# * `Attrition`: There's an imbalance in the data with 16% of the employees attriting and rest not.
# 
# 
# 
# * **Attrition vs Earnings of employee**:
#     * Employees having lower Daily rates and less monthly wage are more likely to attrite.
#     * Monthly rate and the hourly rate doesn't seem to have any effect on attrition.
#     * Lesser salary hike also contributes to attrition.
# 
# * **Attrition vs Years working in company**
#     * Those employees who have spent less time at a company, in a current role, or with a manager have higher chances of attriting.
#     * Training doesn't seem to have an impact on Attrition.
#     
# * **Attrition vs Previous job roles**
#     * Employees who have worked in more companies generally tend to switch more jobs hence attriting.
#     * Employees who attrite generally have lesser years of experience.
# 
# *  **Attrition vs Department, Job Role and Education**
#     * Majority of the employees work for the R&D department. The probability of attrition is least there.
#     * Both Sales and HR have similar Attrition probability.
#     * Sales Executives have an attrition probability of >40%.
#     * Laboratory Technicians and Human Resource personnel also have high probabilities of attrition.
#     * Attrition probability among Research Directors, Manufacturing directors Healthcare representatives, and Managers are much lower than the average attrition probability of 16%.
#     * 25% of employees educated in human resources attrite.
#     * Attrition probability is also high in the case of marketing and technical degree holders.
# 
# 
# * Employees who say they have low satisfaction with their work environments are likely to attrite.
# * There's a ~40% probability of attrition among employees with low ratings for environment satisfaction.
# 
# * Job Involvement looks like a very strong indicator of attrition.
# * Higher the job involvement, greater is the chance that the employee will stay with us and not attrite.
# * Employees unhappy with their job involvement have ~55% probability of attriting (those who rated 0 and 1).
# * Further investigation to understand how this variable was collected will give more insights.
# 
# * Salary hikes are a function of Performance ratings.
# * We have to investigate why the employees who get Excellent(3) and Outstanding(4) Performance rating attrite and how then can they be retained.

# ### To jump back to the EDA summary section, click <a href = #link2>here</a>.
