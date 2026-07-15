"""
Job Change Prediction — Neural Network Optimization

A binary classification case study comparing optimizer, regularization,
and initialization strategies for a feed-forward neural network.
"""

# # Job Change Prediction — Neural Network Optimization
# 
# *A binary classification case study comparing optimizer, regularization, and initialization strategies for a feed-forward neural network.*

# <center><p float="center">
#   <img src="https://images.pexels.com/photos/845451/pexels-photo-845451.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2" width="720"/>
# </p></center>

# ## Problem Statement

# ### Context and Objective

# A data science training provider wants to predict which of its trainees are likely to look for a new job after completing a course, versus which are likely to stay with the training company itself. Knowing this in advance helps the company plan hiring, focus retention efforts, and reduce wasted training spend on candidates who are likely to leave.
# 
# Using demographic, education, and work-experience data collected at signup, this project builds and compares several neural network configurations to predict the probability that a given trainee is looking for a job change, and interprets which factors most influence that decision.

# ### Data Description

# * **Enrollee_id:** Unique ID for candidate
# * **City:** City code
# * **City_development_index:** Developement index of the city (scaled)
# * **Gender:** Gender of candidate
# * **Relevent_experience:** Relevant experience of candidate
# * **Enrolled_university:** Type of University course enrolled if any
# * **Education_level:** Education level of candidate
# * **Major_discipline:** Education major discipline of candidate
# * **Experience:** Candidate total experience in years
# * **Company_size:** No of employees in current employer's company
# * **Company_type:** Type of current employer
# * **Last_new_job:** Difference in years between previous job and current job
# * **Training_hours:** training hours completed
# * **Target:** 0 – Not looking for job change, 1 – Looking for a job change

# ## Installing and Importing the Necessary Libraries

# Install dependencies (run once):
# pip install tensorflow==2.15.0 scikit-learn==1.2.2 matplotlib===3.7.1 seaborn==0.13.1 numpy==1.25.2 pandas==1.5.3 -q --user

# Note: After running the above cell, kindly restart the notebook kernel and run all cells sequentially from the below again.

# Library for data manipulation and analysis.
import pandas as pd
# Fundamental package for scientific computing.
import numpy as np
#splitting datasets into training and testing sets.
from sklearn.model_selection import train_test_split
#Imports tools for data preprocessing including label encoding, one-hot encoding, and standard scaling
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler
#Imports a class for imputing missing values in datasets.
from sklearn.impute import SimpleImputer
#Imports the Matplotlib library for creating visualizations.
import matplotlib.pyplot as plt
# Imports the Seaborn library for statistical data visualization.
import seaborn as sns
# Time related functions.
import time
#Imports functions for evaluating the performance of machine learning models
from sklearn.metrics import confusion_matrix, f1_score,accuracy_score, recall_score, precision_score, classification_report


#Imports the tensorflow,keras and layers.
import tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dense, Input, Dropout,BatchNormalization
from tensorflow.keras import backend

# to suppress unnecessary warnings
import warnings
warnings.filterwarnings("ignore")

# Set the seed using keras.utils.set_random_seed. This will set:
# 1) `numpy` seed
# 2) backend random seed
# 3) `python` random seed
tf.keras.utils.set_random_seed(812)

# If using TensorFlow, this will make GPU ops as deterministic as possible,
# but it will affect the overall performance, so be mindful of that.
tf.config.experimental.enable_op_determinism()

# ## Loading the Data

#Reading the dataset.
Data = pd.read_csv('Data.csv')

# ## Data Overview

# ### Displaying the first few rows of the dataset

# Let's view the first 5 rows of the data
Data.head()

# ### Displaying the last few rows of the dataset

# Let's view the last 5 rows of the data
Data.tail()

# ### Checking the shape of the dataset

# Checking the number of rows and columns in the data
Data.shape

# * The dataset has 19158 rows and 14 columns

# ### Checking the data types of the columns of the dataset

# Let's check the datatypes of the columns in the dataset
Data.info()

# * There are 19,158  observations and 14 columns in the data.
# * 10 columns are of the object datatype and 4 columns are numerical.

# ### Checking for duplicate values

# Let's check for duplicate values in the data
Data.duplicated().sum()

# Let's check for missing values in the data
round(Data.isnull().sum() / Data.isnull().count() * 100, 2)

Data["Target"].value_counts(1)

# Let's view the statistical summary of the numerical columns in the data
Data.describe().T

# Outside of the Enrollee_id (an ID column) and the Target variable, there are only two numerical columns in the dataset.
# 
# * The maximum number of training hours is 336, but at least 75% of employees had finished their training within 88 hours.
# * 50% of the rows have a city development index between 0.903 and 0.949, so the dataset is weighted towards the higher side with respect to this column.

# Let's check the number of unique values in each column
Data.nunique()

# * Each value of the column 'Employee_id' is a unique identifier for an employee. Hence we can drop this column as it will not add any predictive power or value to the model.
# * The 'City' column has 123 unique categories.

for i in Data.describe(include=["object"]).columns:
    print("Unique values in", i, "are :")
    print(Data[i].value_counts())
    print("*" * 50)

# * The 'City' column has 123 unique categories, and the city with the highest number of employees is City 103.
# * Over 90% of the employees in this dataset are males, so it is highly gender-skewed.
# * Most of the employees (~70%) have relevant experience in Data Science.
# * 70% of the employees did not enroll in any of the courses.
# * Most of the employees have a Bachelor's level of education (Graduation) but not more than that (~62% of the total number of employees). There are very few employees with Master's degrees and PhDs.
# * Almost all the employees have previous experience (Related and Non-Related).
# * ~75% of the employees are from private companies.

# ## <a name='link2'>Exploratory Data Analysis (EDA) Summary</a>

# ### **Note**: The EDA section has been covered in detail in the previous case studies. In this case study, we will mainly focus on the model building aspects. We will only be looking at the key observations from EDA. The detailed EDA can be found in the <a href = #link1>appendix section</a>.

# **The below functions need to be defined to carry out the Exploratory Data Analysis.**

# Function to create labeled barplots


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
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n].sort_values(),
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

# Function to plot a boxplot and a histogram along the same scale.


def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (12,7))
    kde: whether to the show density curve (default False)
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
    )  # boxplot will be created and a star will indicate the mean value of the column
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter"
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    )  # For histogram
    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram

### Function to plot distributions


def distribution_plot_wrt_target(data, predictor, target):

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    target_uniq = data[target].unique()

    axs[0, 0].set_title("Distribution of target for target=" + str(target_uniq[0]))
    sns.histplot(
        data=data[data[target] == target_uniq[0]],
        x=predictor,
        kde=True,
        ax=axs[0, 0],
        color="teal",
    )

    axs[0, 1].set_title("Distribution of target for target=" + str(target_uniq[1]))
    sns.histplot(
        data=data[data[target] == target_uniq[1]],
        x=predictor,
        kde=True,
        ax=axs[0, 1],
        color="orange",
    )

    axs[1, 0].set_title("Boxplot w.r.t target")
    sns.boxplot(data=data, x=target, y=predictor, ax=axs[1, 0], palette="gist_rainbow")

    axs[1, 1].set_title("Boxplot (without outliers) w.r.t target")
    sns.boxplot(
        data=data,
        x=target,
        y=predictor,
        ax=axs[1, 1],
        showfliers=False,
        palette="gist_rainbow",
    )

    plt.tight_layout()
    plt.show()

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

# ### Univariate Analysis

# #### Gender

labeled_barplot(Data, "Gender",perc=True)

# * Over 90% of this dataset is male, representing a highly gender-skewed dataset. This could be a limitation with respect to implementing this model in the real world, since gender balance is highly important to create machine learning models that are practically implemented on datasets related to people.

# #### Relevant Experience

labeled_barplot(Data, "Relevent_experience",perc=True)

# * Most of the employees have relevant prior experience (~70%).
# * 30% of the employees, however, have no relevant experience.

# #### Education level

labeled_barplot(Data, "Education_level",perc=True)

# * Approximately 62% of employees have a Bachelor's (Graduate) level of education, but not more than that.
# * Approx 23% of employees have a Master's degree as their highest level of education.
# * There are very few employees (~1.5%) with only a High School level of education or below.

# #### Company type

labeled_barplot(Data, "Company_type",perc=True)

# * Approximately 52% of the total employees are from a private limited company, showing the skew of the profile towards the private sector.

# #### Target

labeled_barplot(Data,"Target",perc=True)

# * This bar chart shows that the actual distribution of classes is itself imbalanced for the target variable.
# * Only ~25% of the employees are actually looking for a job change.
# 
# Hence, this dataset and problem statement represent an example of Imbalanced Classification.

# ### Bivariate Analysis

# #### Gender and Target

stacked_barplot(Data, "Gender", "Target")

# * From the above plot, it is observed that the likelihood of the employee choosing a job switch does not depend on their gender.

# #### Relevant experience and Target

stacked_barplot(Data, "Relevent_experience", "Target")

# * From the above plot, we see that employees from Non-relevant experience are more likely to be switching their job.

# #### Education level and Target

stacked_barplot(Data, "Education_level", "Target")

# * Employees who completed Graduation and Master's degrees are more likely to be trying to switch their jobs.

# #### Company type and Target

stacked_barplot(Data, "Company_type", "Target")

# * From the above plot, it is observed that the likelihood of the employee choosing a job switch does not depend on the type of the company.

# ## Data Preprocessing

# ### Column Binning

# - The column "City_development_index" serves as a proxy to the column "City".  Hence, we can drop the column named "City" and retain "City_development_index"

###Dropping the column as they will not add value to the modeling
Data.drop(['City'], axis=1, inplace=True)

# - Binning the "Experience" variable into 5 bins

Data["Experience"].unique()

# The
# below  function categorizes numerical values into predefined bins. It checks the value of `x` against specific ranges and returns a corresponding category. If `x` falls within a certain range, it returns a string representing that range. If `x` does not fall into any predefined range, it simply returns `x` itself.

def bin(x):
    if x=='>20':
      return '>20'
    elif x in [str(i) for i in range(16,21)]:
      return '16-20'
    elif x in [str(i) for i in range(11,16)]:
      return '10-15'
    elif x in [str(i) for i in range(5,11)]:
      return '6-10'
    elif x in [str(i) for i in range(1,6)] + ['<1']:
      return '<6'
    else:
      return x

Data["Experience"] = Data["Experience"].apply(bin)

Data["Experience"].unique()

# ID column consists of uniques ID for clients and hence will not add value to the modeling
Data.drop(columns="Enrollee_id", inplace=True)

# ### Missing Value Imputation

## Separating Independent and Dependent Columns
X = Data.drop(['Target'],axis=1)
Y = Data['Target']

X.columns

#Calculating the total number of nan values for each columns.
X.isnull().sum()

# - There are no missing values in the numerical columns.

# * Hence, we will impute the missing values in the categorical columns only using their mode.

imputer_mode = SimpleImputer(strategy="most_frequent")
X[["Enrolled_university","Education_level","Major_discipline","Experience","Company_type","Last_new_job","Company_size","Gender"]] = imputer_mode.fit_transform(
    X[["Enrolled_university","Education_level","Major_discipline","Experience","Company_type","Last_new_job","Company_size","Gender"]])

# ### Encoding the categorical variables

# Encoding the categorical variables using one-hot encoding
X = pd.get_dummies(
    X,
    columns=["Last_new_job","Relevent_experience","Enrolled_university","Education_level","Major_discipline","Experience","Company_type","Company_size","Gender"],
    drop_first=True,
)

X = X.astype(float)

# ### Normalizing the numerical variables

#Standardizing the numerical variables to zero mean and unit variance.
transformer = StandardScaler()
X[["City_development_index","Training_hours"]] = transformer.fit_transform(X[["City_development_index","Training_hours"]])

# ### Splitting the dataset

# Splitting the dataset into the Training and Test set.
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42,stratify = Y)

# Splitting the Train dataset into the Training and Validation set.
X_train, X_valid, y_train, y_valid = train_test_split(X_train,y_train, test_size = 0.2, random_state = 42,stratify = y_train)

#Printing the shapes.
print(X_train.shape,y_train.shape)
print(X_valid.shape,y_valid.shape)
print(X_test.shape,y_test.shape)

y_train = y_train.to_numpy()
y_valid = y_valid.to_numpy()
y_test = y_test.to_numpy()

# ### Utility functions

def plot(history, name):
    """
    Function to plot loss/accuracy

    history: an object which stores the metrics and losses.
    name: can be one of Loss or Accuracy
    """
    fig, ax = plt.subplots() #Creating a subplot with figure and axes.
    plt.plot(history.history[name]) #Plotting the train accuracy or train loss
    plt.plot(history.history['val_'+name]) #Plotting the validation accuracy or validation loss

    plt.title('Model ' + name.capitalize()) #Defining the title of the plot.
    plt.ylabel(name.capitalize()) #Capitalizing the first letter.
    plt.xlabel('Epoch') #Defining the label for the x-axis.
    fig.legend(['Train', 'Validation'], loc="outside right upper") #Defining the legend, loc controls the position of the legend.

# defining a function to compute different metrics to check performance of a classification model built using statsmodels
def model_performance_classification(
    model, predictors, target, threshold=0.5
):
    """
    Function to compute different metrics to check classification model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    threshold: threshold for classifying the observation as class 1
    """

    # checking which probabilities are greater than threshold
    pred = model.predict(predictors) > threshold
    # pred_temp = model.predict(predictors) > threshold
    # # rounding off the above values to get classes
    # pred = np.round(pred_temp)

    acc = accuracy_score(target, pred)  # to compute Accuracy
    recall = recall_score(target, pred, average='weighted')  # to compute Recall
    precision = precision_score(target, pred, average='weighted')  # to compute Precision
    f1 = f1_score(target, pred, average='weighted')  # to compute F1-score

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {"Accuracy": acc, "Recall": recall, "Precision": precision, "F1 Score": f1,},
        index=[0],
    )

    return df_perf

# ## Model Building

# ### Model Evaluation Criteria

# **A model can make wrong predictions in the following ways:**
# * Predicting an employee is looking for a job, when he/she is not looking for it.
# * Predicting an employee is not looking for a job, when he/she is in fact looking for one.
# 
# **Which case is more important?**
# 
# Both cases are actually important for the purposes of this case study. Not giving a chance to a deserving employee (by wrongly classifying them as likely to attrite) might lead to decreased productivity, and the company might lose a good employee affecting the organization's growth. However, giving chances to a non-deserving employee (as they are likely to attrite) would lead to a financial loss for the company, and giving such employees an increased amount of responsibility might again affect the company's growth.
# 
# **How to reduce this loss i.e need to reduce False Negatives as well as False Positives?**
# 
# Since both errors are important for us to minimize, the company would want the F1 Score evaluation metric to be maximized/ Hence, the focus should be on increasing the F1 score rather than focusing on just one metric i.e. Recall or Precision.

# As we have are dealing with an imbalance in class distribution, we will be using class weights to allow the model to give proportionally more importance to the minority class.

# Calculate class weights for imbalanced dataset
cw = (y_train.shape[0]) / np.bincount(y_train)

# Create a dictionary mapping class indices to their respective class weights
cw_dict = {}
for i in range(cw.shape[0]):
    cw_dict[i] = cw[i]

cw_dict

# defining the batch size and # epochs upfront as we'll be using the same values for all models
epochs = 25
batch_size = 64

# ### Model 0

# - Let's start with a neural network consisting of
#   - two hidden layers with 14 and 7 neurons respectively
#   - activation function of ReLU.
#   - SGD as the optimizer

# clears the current Keras session, resetting all layers and models previously created, freeing up memory and resources.
tf.keras.backend.clear_session()

#Initializing the neural network
model = Sequential()
model.add(Dense(14,activation="relu",input_dim=X_train.shape[1]))
model.add(Dense(7,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

model.summary()

optimizer = tf.keras.optimizers.SGD()    # defining SGD as the optimizer to be used
model.compile(loss='binary_crossentropy', optimizer=optimizer)

start = time.time()
history = model.fit(X_train, y_train, validation_data=(X_valid,y_valid) , batch_size=batch_size, epochs=epochs,class_weight=cw_dict)
end=time.time()

print("Time taken in seconds ",end-start)

plot(history,'loss')

model_0_train_perf = model_performance_classification(model, X_train, y_train)
model_0_train_perf

model_0_valid_perf = model_performance_classification(model, X_valid, y_valid)
model_0_valid_perf

# - Train F1 score of ~0.73 and test F1 score of ~0.71 indicate consistent performance of the model between training and testing datasets.
# 
# - Even though it's a good score, the rate of improvement over the epochs is still low.

# ### Model 1

# - After the 5th epoch, the model's rate of learning is low.
# - Let's try adding momentum to check whether it's accelerating the learning process.

# clears the current Keras session, resetting all layers and models previously created, freeing up memory and resources.
tf.keras.backend.clear_session()

#Initializing the neural network
model = Sequential()
model.add(Dense(14,activation="relu",input_dim=X_train.shape[1]))
model.add(Dense(7,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

model.summary()

optimizer = tf.keras.optimizers.SGD(momentum=0.9)    # defining SGD as the optimizer to be used
model.compile(loss='binary_crossentropy', optimizer=optimizer)

start = time.time()
history = model.fit(X_train, y_train, validation_data=(X_valid,y_valid) , batch_size=batch_size, epochs=epochs,class_weight = cw_dict)
end=time.time()

print("Time taken in seconds ",end-start)

plot(history,'loss')

model_1_train_perf = model_performance_classification(model, X_train, y_train)
model_1_train_perf

model_1_valid_perf = model_performance_classification(model, X_valid, y_valid)
model_1_valid_perf

# - As expected, there is a improvement in the train scores.
# - But, the difference between the train and valid scores have increased.

# ### Model 2

# - Let's change the optimizer to Adam
#     - This will introduce momentum as well as an adaptive learning rate

# clears the current Keras session, resetting all layers and models previously created, freeing up memory and resources.
tf.keras.backend.clear_session()

#Initializing the neural network
model = Sequential()
model.add(Dense(14,activation="relu",input_dim=X_train.shape[1]))
model.add(Dense(7,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

model.summary()

optimizer = tf.keras.optimizers.Adam()    # defining Adam as the optimizer to be used
model.compile(loss='binary_crossentropy', optimizer=optimizer)

start = time.time()
history = model.fit(X_train, y_train, validation_data=(X_valid,y_valid) , batch_size=batch_size, epochs=epochs,class_weight=cw_dict)
end=time.time()

print("Time taken in seconds ",end-start)

plot(history,'loss')

model_2_train_perf = model_performance_classification(model, X_train, y_train)
model_2_train_perf

model_2_valid_perf = model_performance_classification(model, X_valid, y_valid)
model_2_valid_perf

# - The difference between the train and valid scores have decreased but not to a great extent.

# ### Model 3

# - The difference between the train loss and test loss is high.
# - Let's add dropout to regularize it.

# clears the current Keras session, resetting all layers and models previously created, freeing up memory and resources.
tf.keras.backend.clear_session()

#Initializing the neural network
model = Sequential()
model.add(Dense(14,activation="relu",input_dim=X_train.shape[1]))
model.add(Dropout(0.4))
model.add(Dense(7,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1,activation="sigmoid"))

model.summary()

optimizer = tf.keras.optimizers.Adam()    # defining Adam as the optimizer to be used
model.compile(loss='binary_crossentropy', optimizer=optimizer)

start = time.time()
history = model.fit(X_train, y_train, validation_data=(X_valid,y_valid) , batch_size=batch_size, epochs=epochs,class_weight=cw_dict)
end=time.time()

print("Time taken in seconds ",end-start)

plot(history,'loss')

model_3_train_perf = model_performance_classification(model, X_train, y_train)
model_3_train_perf

model_3_valid_perf = model_performance_classification(model, X_valid, y_valid)
model_3_valid_perf

# - The difference between train and validation scores has still not reduced.

# ### Model 4

# - The scores are still not that good.
# - Let's add batch normalization to see whether we can stabilize the training process and thereby improve the scores.

# clears the current Keras session, resetting all layers and models previously created, freeing up memory and resources.
tf.keras.backend.clear_session()

#Initializing the neural network
model = Sequential()
model.add(Dense(14,activation="relu",input_dim=X_train.shape[1]))
model.add(BatchNormalization())
model.add(Dense(7,activation="relu"))
model.add(BatchNormalization())
model.add(Dense(1,activation="sigmoid"))

model.summary()

optimizer = tf.keras.optimizers.Adam()    # defining Adam as the optimizer to be used
model.compile(loss='binary_crossentropy', optimizer=optimizer)

start = time.time()
history = model.fit(X_train, y_train, validation_data=(X_valid,y_valid) , batch_size=batch_size, epochs=epochs,class_weight=cw_dict)
end=time.time()

print("Time taken in seconds ",end-start)

plot(history,'loss')

model_4_train_perf = model_performance_classification(model, X_train, y_train)
model_4_train_perf

model_4_valid_perf = model_performance_classification(model, X_valid, y_valid)
model_4_valid_perf

# - There's no improvement in the scores.

# ### Model 5

# - Let's add both batchnormalization and dropout.

# clears the current Keras session, resetting all layers and models previously created, freeing up memory and resources.
tf.keras.backend.clear_session()

#Initializing the neural network
model = Sequential()
model.add(Dense(14,activation="relu",input_dim=X_train.shape[1]))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(7,activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1,activation="sigmoid"))

model.summary()

optimizer = tf.keras.optimizers.Adam()    # defining Adam as the optimizer to be used
model.compile(loss='binary_crossentropy', optimizer=optimizer)

start = time.time()
history = model.fit(X_train, y_train, validation_data=(X_valid,y_valid) , batch_size=batch_size, epochs=epochs,class_weight=cw_dict)
end=time.time()

print("Time taken in seconds ",end-start)

plot(history,'loss')

model_5_train_perf = model_performance_classification(model, X_train, y_train)
model_5_train_perf

model_5_valid_perf = model_performance_classification(model, X_valid, y_valid)
model_5_valid_perf

# - There's no improvement in the scores, but there has been a slight decrease in the difference between train and validation scores.

# ### Model 6

# - Let's initialize the weights using He normal.
# - We'll also use only Dropout for regularization.

# clears the current Keras session, resetting all layers and models previously created, freeing up memory and resources.
tf.keras.backend.clear_session()

#Initializing the neural network
model = Sequential()
model.add(Dense(14,activation="relu",kernel_initializer="he_normal",input_dim=X_train.shape[1]))
model.add(Dropout(0.4))
model.add(Dense(7,activation="relu",kernel_initializer="he_normal"))
model.add(Dropout(0.2))
model.add(Dense(1,activation="sigmoid",kernel_initializer="he_normal"))

model.summary()

optimizer = tf.keras.optimizers.Adam()    # defining Adam as the optimizer to be used
model.compile(loss='binary_crossentropy', optimizer=optimizer)

start = time.time()
history = model.fit(X_train, y_train, validation_data=(X_valid,y_valid) , batch_size=batch_size, epochs=epochs,class_weight=cw_dict)
end=time.time()

print("Time taken in seconds ",end-start)

plot(history,'loss')

model_6_train_perf = model_performance_classification(model, X_train, y_train)
model_6_train_perf

model_6_valid_perf = model_performance_classification(model, X_valid, y_valid)
model_6_valid_perf

# - There's a slight improvement in the scores.
# - The difference between train and validation scores has also reduced.

# ## Model Performance Comparison and Final Model Selection

# training performance comparison

models_train_comp_df = pd.concat(
    [
        model_0_train_perf.T,
        model_1_train_perf.T,
        model_2_train_perf.T,
        model_3_train_perf.T,
        model_4_train_perf.T,
        model_5_train_perf.T,
        model_6_train_perf.T
    ],
    axis=1,
)
models_train_comp_df.columns = [
    "Neural Network (SGD, No Regularization)",
    "Neural Network (SGD with Momentum, No Regularization)",
    "Neural Network (Adam , No Regularization)",
    "Neural Network (Adam, dropout [0.4,0.2])",
    "Neural Network (Adam, Batch Normalization)",
    "Neural Network (dropout [0.4,0.2], Batch Normalization)",
    "Neural Network (Adam,dropout [0.4,0.2] ,He initialization)"
]

#Validation performance comparison

models_valid_comp_df = pd.concat(
    [
        model_0_valid_perf.T,
        model_1_valid_perf.T,
        model_2_valid_perf.T,
        model_3_valid_perf.T,
        model_4_valid_perf.T,
        model_5_valid_perf.T,
        model_6_valid_perf.T
    ],
    axis=1,
)
models_valid_comp_df.columns = [
    "Neural Network (SGD, No Regularization)",
    "Neural Network (SGD with Momentum, No Regularization)",
    "Neural Network (Adam , No Regularization)",
    "Neural Network (Adam, dropout [0.4,0.2])",
    "Neural Network (Adam, Batch Normalization)",
    "Neural Network (dropout [0.4,0.2], Batch Normalization)",
    "Neural Network (Adam,dropout [0.4,0.2] ,He initialization)"
]

models_train_comp_df

models_valid_comp_df

models_train_comp_df.loc["F1 Score"] - models_valid_comp_df.loc["F1 Score"]

# - Final Model: **Neural Network (Adam,dropout [0.4,0.2] ,He initialization)**
# - Reasoning:
#   - Best F1 score on the training set (~0.74), indicating strong performance in learning from the training data.
#   - Best F1 score on the validation set (~0.72), demonstrating good performance in generalizing to unseen data.
#   - When considering validation scores only, Model 6 outperforms others, suggesting its effectiveness in real-world applications and new data scenarios.

# ### Final Model

# clears the current Keras session, resetting all layers and models previously created, freeing up memory and resources.
tf.keras.backend.clear_session()

#Initializing the neural network
model = Sequential()
model.add(Dense(14,activation="relu",kernel_initializer="he_normal",input_dim=X_train.shape[1]))
model.add(Dropout(0.4))
model.add(Dense(7,activation="relu",kernel_initializer="he_normal"))
model.add(Dropout(0.2))
model.add(Dense(1,activation="sigmoid",kernel_initializer="he_normal"))

model.summary()

optimizer = tf.keras.optimizers.Adam()    # defining Adam as the optimizer to be used
model.compile(loss='binary_crossentropy', optimizer=optimizer)

start = time.time()
history = model.fit(X_train, y_train, validation_data=(X_valid,y_valid) , batch_size=batch_size, epochs=epochs,class_weight=cw_dict)
end=time.time()

y_train_pred = model.predict(X_train)
y_valid_pred = model.predict(X_valid)
y_test_pred = model.predict(X_test)

print("Classification Report - Train data",end="\n\n")
cr = classification_report(y_train,y_train_pred>0.5)
print(cr)

print("Classification Report - Validation data",end="\n\n")
cr = classification_report(y_valid,y_valid_pred>0.5)
print(cr)

print("Classification Report - Test data",end="\n\n")
cr = classification_report(y_test,y_test_pred>0.5)
print(cr)

# - The weighted F1 score on the test data is ~0.74
# 
# - An F1 score of ~0.74 indicates a good balance between precision and recall, suggesting moderate performance in accurately classifying instances with minimal false positives and false negatives.
# 
# - Model can be further tuned to deal with minority class.

# ## Business Insights and Recommendations

# * The HR department of the company can deploy the final model from this exercise to identify with a reasonable degree of accuracy whether an employee is likely to switch jobs or not, and this process seems to be easier and more time-efficient than other methods.
# 
# - The company should prioritize gender diversity initiatives in its recruitment strategies to create a more balanced and inclusive workforce. Diverse teams are known to foster innovation and bring varied perspectives, which can ultimately benefit the company's performance.
# 
# - Offer upskilling or reskilling programs to bridge the gap between employees' skills and job requirements, thereby improving job satisfaction and reducing the likelihood of attrition. Additionally, provide career advancement pathways for employees with diverse backgrounds to retain talent and foster a culture of continuous learning and growth.
# 
# - Consider offering incentives or opportunities for career growth and development to employees in cities with higher development indices to improve retention rates.
# 
# - One can employ techniques like oversampling minority classes, using appropriate evaluation metrics (e.g., F1-score), or employing ensemble methods like SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance. Ensuring a balanced dataset is crucial for model training to avoid biased predictions.

# ## <a name='link1'>Appendix: Detailed Exploratory Data Analysis (EDA)</a>

# ### Univariate Analysis

# #### City Development Index

histogram_boxplot(Data, "City_development_index")

# * From the above plot, we observe that there are many people from cities having a development index more than 0.9.

# #### Training Hours

histogram_boxplot(Data, "Training_hours")

# * From the plot, we observe that the measures of central tendency with respect to training hours seem to be 70, despite a maximum value over 300 hours. So most of the people in this dataset have undergone traning for less than 100 hours.

# #### Enrolled University

labeled_barplot(Data, "Enrolled_university",perc=True)

# * 72% of the employees did not enroll in any of the courses.
# * Approximately 20% of the employees have enrolled themselves in full-time courses.
# * Only 6% have enrolled in part-time courses.

# #### Major discipline

labeled_barplot(Data, "Major_discipline",perc=True)

# * Approximately 75% of employees have opted for STEM as their major discipline.

# #### Experience

labeled_barplot(Data, "Experience",perc=True)

# * Approximately 17% of total employees have over 20 years of work experience.

# ### Bivariate Analysis

# #### City development index and Target

distribution_plot_wrt_target(Data, "City_development_index", "Target")

# * From the above plot, we observe that employees from cities having a development index over 0.9, are not willing to switch their jobs.

# #### Training hours and Target

distribution_plot_wrt_target(Data, "Training_hours", "Target")

# * We observe that the distribution of the training hours with respect to the target variable is rightly skewed, and from the box plot for both classes the median traning hours are around 50.

# #### Enrolled university and Target

stacked_barplot(Data, "Enrolled_university", "Target")

# * Employees who have taken full-time courses in universities are the ones who are more likely to be trying to switch jobs.

# #### Major discipline and Target

stacked_barplot(Data, "Major_discipline", "Target")

# * Employees who took STEM or Business Degrees as their major discipline are slightly more likely to change their job.

# #### Experience and Target

stacked_barplot(Data, "Experience", "Target")

# * From the above plot, it's clear that employees having a work experience of less than 3 years are trying to switch their jobs.

# #### Last_new_job and Target

stacked_barplot(Data, "Last_new_job", "Target")

# * Employees who have never switched their job before are the most likely to be looking for a job change.

# ### To jump back to the EDA summary section, click <a href = #link2>here</a>.

# <font size=5 color='blue'>Power Ahead!</font>
# ___
