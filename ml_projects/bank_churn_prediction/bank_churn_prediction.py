"""
Bank Churn Prediction — Neural Networks

Predicts customer churn using a series of tuned neural network
architectures (SGD, Adam, dropout, SMOTE for class imbalance),
optimizing for recall to minimize missed churn cases.
"""

# # Bank Churn Prediction — Neural Networks
# 
# *Predicting customer churn using a series of tuned neural network architectures, addressing class imbalance with SMOTE.*

# ## Problem Statement

# ### Context

# Businesses like banks which provide service have to worry about problem of 'Customer Churn' i.e. customers leaving and joining another service provider. It is important to understand which aspects of the service influence a customer's decision in this regard. Management can concentrate efforts on improvement of service, keeping in mind these priorities.

# ### Objective

# You as a Data scientist with the  bank need to  build a neural network based classifier that can determine whether a customer will leave the bank  or not in the next 6 months.

# ### Data Dictionary

# * CustomerId: Unique ID which is assigned to each customer
# 
# * Surname: Last name of the customer
# 
# * CreditScore: It defines the credit history of the customer.
#   
# * Geography: A customer’s location
#    
# * Gender: It defines the Gender of the customer
#    
# * Age: Age of the customer
#     
# * Tenure: Number of years for which the customer has been with the bank
# 
# * NumOfProducts: refers to the number of products that a customer has purchased through the bank.
# 
# * Balance: Account balance
# 
# * HasCrCard: It is a categorical variable which decides whether the customer has credit card or not.
# 
# * EstimatedSalary: Estimated salary
# 
# * isActiveMember: Is is a categorical variable which decides whether the customer is active member of the bank or not ( Active member in the sense, using bank products regularly, making transactions etc )
# 
# * Exited : whether or not the customer left the bank within six month. It can take two values
# ** 0=No ( Customer did not leave the bank )
# ** 1=Yes ( Customer left the bank )

# Install dependencies (run once):
# # Installing the libraries with the specified version.
# pip install tensorflow==2.15.0 scikit-learn==1.2.2 seaborn==0.13.1 matplotlib==3.7.1 numpy==1.25.2 pandas==2.0.3 imbalanced-learn==0.10.1 -q --user

# ## Importing necessary libraries

# Libraries to help with reading and manipulating data
import pandas as pd
import numpy as np

# libaries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Library to split data
from sklearn.model_selection import train_test_split

# library to import to standardize the data
from sklearn.preprocessing import StandardScaler, LabelEncoder

# importing different functions to build models
import tensorflow as tf
from tensorflow import keras
from keras import backend
from keras.models import Sequential
from keras.layers import Dense, Dropout

# importing SMOTE
from imblearn.over_sampling import SMOTE

# importing metrics
from sklearn.metrics import confusion_matrix,roc_curve,classification_report,recall_score

import random

# Library to avoid the warnings
import warnings
warnings.filterwarnings("ignore")

# ## Loading the dataset

# Colab-specific data loading, replace with a local path outside Colab:
# from google.colab import drive
# drive.mount('/content/drive')
ds = pd.read_csv("bank-1.csv")

# ## Data Overview

# let's view the first 5 rows of the data
ds.head()

# let's view the last 5 rows of the data
ds.tail()

# Checking the number of rows and columns in the training data
ds.shape

# Checking the data types of columns
ds.info()

# Checking the Statistical summary
ds.describe().T

# let's check for missing values in the data
ds.isnull().sum()

# Checking for unique values in dataset
ds.nunique()

#RowNumber , CustomerId and Surname are unique hence dropping it
ds = ds.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# ## Exploratory Data Analysis

# ### Univariate Analysis

# function to plot a boxplot and a histogram along the same scale.


def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (12,7))
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

# **Observations on Credit Score**

histogram_boxplot(ds,'CreditScore')          ## Creating histogram_boxplot for CreditScore

# **Observations on Age**

histogram_boxplot(ds,'Age')          ## Creating histogram_boxplot for Age

# **Observations on Balance**

histogram_boxplot(ds,'Balance')          ## Code to create histogram_boxplot for Balance

# **Observations on Estimated Salary**

histogram_boxplot(ds,'EstimatedSalary')          ## Code to create histogram_boxplot for Balance

# **Observations on Exited**

labeled_barplot(ds, "Exited", perc=True)          ## Code to create labeled_barplot for Exited

# **Observations on Geography**

labeled_barplot(ds, 'Geography', perc=True)          ## Code to create labeled_barplot for Geography

# **Observations on Gender**

labeled_barplot(ds, 'Gender', perc=True)               ## Code to create labeled_barplot for Gender

# **Observations on Tenure**

labeled_barplot(ds, 'Tenure', perc=True)

# **Observations on Number of Products**

labeled_barplot(ds,'NumOfProducts', perc=True)

# **Observations on has Credit Card**

labeled_barplot(ds, 'HasCrCard', perc=True)

# **Observations on is Active Member**

labeled_barplot(ds, 'IsActiveMember', perc=True)

# ### Bivariate Analysis

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

# **Correlation Plot**

# defining the list of numerical columns
cols_list = ["CreditScore","Age","Tenure","Balance","EstimatedSalary"]

plt.figure(figsize=(15, 7))
sns.heatmap(ds[cols_list].corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral")
plt.show()

# **Exited Vs Geography**

stacked_barplot(ds, "Geography", "Exited" )

# **Exited Vs Gender**

stacked_barplot(ds, "Gender", "Exited" )

# **Exited Vs Has Credit Card**

stacked_barplot(ds, "HasCrCard", "Exited" )

# **Exited Vs Is Active Member**

stacked_barplot(ds, "IsActiveMember", "Exited" )

# **Exited Vs Credit Score**

plt.figure(figsize=(5,5))
sns.boxplot(y='CreditScore',x='Exited',data=ds)
plt.show()

# **Exited Vs Age**

plt.figure(figsize=(5,5))
sns.boxplot(y='Age',x='Exited',data=ds)
plt.show()

# **Exited Vs Tenure**

plt.figure(figsize=(5,5))
sns.boxplot(y='Tenure',x='Exited',data=ds)
plt.show()

# **Exited Vs Balance**

plt.figure(figsize=(5,5))
sns.boxplot(y='Balance',x='Exited',data=ds)
plt.show()

# **Exited Vs No of Products**

plt.figure(figsize=(5,5))
sns.boxplot(y='NumOfProducts',x='Exited',data=ds)
plt.show()

# **Exited Vs Estimated Salary**

plt.figure(figsize=(5,5))
sns.boxplot(y='EstimatedSalary',x='Exited',data=ds)
plt.show()

# ## Data Preprocessing

# ### Dummy Variable Creation

ds = pd.get_dummies(ds,columns=ds.select_dtypes(include=["object"]).columns.tolist(),drop_first=True)
ds = ds.astype(float)
ds.head()

# ### Train-validation-test Split

X = ds.drop(['Exited'],axis=1) # Credit Score through Estimated Salary
y = ds['Exited'] # Exited

# Splitting the dataset into the Training and Testing set.

X_large, X_test, y_large, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42,stratify=y,shuffle = True)

# Splitting the dataset into the Training and Validation set.

X_train, X_val, y_train, y_val = train_test_split(X_large, y_large, test_size = 0.2, random_state = 42,stratify=y_large, shuffle = True)

print(X_train.shape, X_val.shape, X_test.shape)

print(y_train.shape, y_val.shape, y_test.shape)

# ### Data Normalization

# creating an instance of the standard scaler
sc = StandardScaler()

X_train[cols_list] = sc.fit_transform(X_train[cols_list])
X_val[cols_list] = sc.transform(X_val[cols_list])
X_test[cols_list] = sc.transform(X_test[cols_list])

X_train[cols_list].head()

# ## Model Building

# ### Model Evaluation Criterion

# **Choosing the evaluation metric**
# 
# A bank misses more value by failing to identify a customer who is about to churn (and losing them entirely) than it does by flagging a loyal customer as at-risk (which just costs a bit of unnecessary retention outreach). That makes **recall** the right metric to optimize here — it directly measures how many actual churners the model successfully catches, minimizing missed (false-negative) churn cases.

def make_confusion_matrix(actual_targets, predicted_targets):
    """
    To plot the confusion_matrix with percentages

    actual_targets: actual target (dependent) variable values
    predicted_targets: predicted target (dependent) variable values
    """
    cm = confusion_matrix(actual_targets, predicted_targets)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(cm.shape[0], cm.shape[1])

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

train_metric_df = pd.DataFrame(columns=["recall"])
valid_metric_df = pd.DataFrame(columns=["recall"])

# ### Neural Network with SGD Optimizer

backend.clear_session()
#Fixing the seed for random number generators so that we can ensure we receive the same output everytime
np.random.seed(2)
random.seed(2)
tf.random.set_seed(2)

# Initializing the neural network
model_0 = Sequential()

# Adding the input layer with 64 neurons and relu as activation function
model_0.add(Dense(64, activation='relu', input_dim = X_train.shape[1]))

# Adding a hidden layer with 32 neurons and relu activation
model_0.add(Dense(32, activation='relu'))

# Adding the output layer with 1 neuron for binary classification
model_0.add(Dense(1, activation='sigmoid'))

#Using SGD as the optimizer.
optimizer = tf.keras.optimizers.SGD(0.001)

#Defining Recall metric
metric = keras.metrics.Recall()

model_0.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=[metric])

model_0.summary()

# Fitting the ANN

history_0 = model_0.fit(
    X_train, y_train,
    batch_size=32,    ## Using batch size of 32
    validation_data=(X_val,y_val),
    epochs=20,    ## Specifying the number of epochs = 20
    verbose=1
)

# **Loss Function**

#Plotting Train Loss vs Validation Loss
plt.plot(history_0.history['loss'])
plt.plot(history_0.history['val_loss'])
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# **Recall**

#Plotting Train recall vs Validation recall
plt.plot(history_0.history['recall'])
plt.plot(history_0.history['val_recall'])
plt.title('model recall')
plt.ylabel('Recall')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
#history_0.history

#Predicting the results using best as a threshold
y_train_pred = model_0.predict(X_train)
y_train_pred = (y_train_pred > 0.5)
y_train_pred

y_val_pred = model_0.predict(X_val)    ## Making prediction on the validation set
y_val_pred = (y_val_pred > 0.5)
y_val_pred

model_name = "NN with SGD"

train_metric_df.loc[model_name] = recall_score(y_train, y_train_pred)
valid_metric_df.loc[model_name] = recall_score(y_val, y_val_pred)

# **Classification Report**

#Classification report
cr = classification_report(y_train, y_train_pred)
print(cr)


cr = classification_report(y_val, y_val_pred)    ## Checking the model's performance on the validation set
print(cr)

# **Confusion Matrix**

make_confusion_matrix(y_train, y_train_pred)

make_confusion_matrix(y_val, y_val_pred)

# ## Model Performance Improvement

# ### Neural Network with Adam Optimizer

backend.clear_session()
#Fixing the seed for random number generators so that we can ensure we receive the same output everytime
np.random.seed(2)
random.seed(2)
tf.random.set_seed(2)

# Initializing the neural network
model_1 = Sequential()

# Adding the input layer with 64 neurons and relu as activation function
model_1.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))

# Adding a hidden layer with 32 neurons and relu activation
model_1.add(Dense(64, activation='relu'))

# Adding the output layer with 1 neuron and sigmoid as activation (for binary classification)
model_1.add(Dense(1, activation='sigmoid'))

# Using Adam as the optimizer.
optimizer = tf.keras.optimizers.Adam()

#Defining Recall metric
metric = keras.metrics.Recall()

# Compiling the model with binary cross entropy as loss function and recall as the metric
model_1.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=[metric])

model_1.summary()

#Fitting the ANN
history_1 = model_1.fit(
    X_train,y_train,
    batch_size=32, ## Specifying the batch size of 32
    validation_data=(X_val,y_val),
    epochs=20, ## Using the epochs = 20
    verbose=1
)

# **Loss Function**

#Plotting Train Loss vs Validation Loss
plt.plot(history_1.history['loss'])
plt.plot(history_1.history['val_loss'])
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# **Recall**

#Plotting Train recall vs Validation recall
plt.plot(history_1.history['recall'])
plt.plot(history_1.history['val_recall'])
plt.title('model recall')
plt.ylabel('Recall')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#Predicting the results using 0.5 as the threshold
y_train_pred = model_1.predict(X_train)
y_train_pred = (y_train_pred > 0.5)
y_train_pred

#Predicting the results using 0.5 as the threshold
y_val_pred = model_1.predict(X_val)
y_val_pred = (y_val_pred > 0.5)
y_val_pred

model_name = "NN with Adam"

train_metric_df.loc[model_name] = recall_score(y_train,y_train_pred)
valid_metric_df.loc[model_name] = recall_score(y_val,y_val_pred)

# **Classification Report**

#Classification report
cr=classification_report(y_train,y_train_pred)
print(cr)

# Checking the model's performance on the validation set
cr = classification_report(y_val, y_val_pred)
print(cr)

# **Confusion Matrix**

#Calculating the confusion matrix
make_confusion_matrix(y_train, y_train_pred)

make_confusion_matrix(y_val, y_val_pred)

# ### Neural Network with Adam Optimizer and Dropout

backend.clear_session()
#Fixing the seed for random number generators so that we can ensure we receive the same output everytime
np.random.seed(2)
random.seed(2)
tf.random.set_seed(2)

#Initializing the neural network
model_2 = Sequential()
#Adding the input layer with 32 neurons and relu as activation function
model_2.add(Dense(32,activation='relu',input_dim = X_train.shape[1]))
# Adding dropout with a rate of 0.2
model_2.add(Dropout(0.2))
# Adding a hidden layer with 64 neurons and relu activation
model_2.add(Dense(64,activation='relu'))
# Adding a hidden layer with 32 neurons and relu activation
model_2.add(Dense(32,activation='relu'))
# Adding dropout with a rate of 0.1
model_2.add(Dropout(0.1))
# Adding a hidden layer with 16 neurons and relu activation
model_2.add(Dense(16,activation='relu'))
# Adding the output layer with 1 neuron for binary classification
model_2.add(Dense(1, activation = 'sigmoid'))

# Using Adam as the optimizer.
optimizer = tf.keras.optimizers.Adam()


metric = keras.metrics.Recall()

# Compiling the model with binary cross entropy as loss function and recall as the metric.
model_2.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[metric])

# Summary of the model
model_2.summary()

#Fitting the ANN with batch_size = 32 and 100 epochs
history_2 = model_2.fit(
    X_train,y_train,
    batch_size=32,
    epochs=100,
    verbose=1,
    validation_data=(X_val,y_val)
)

# **Loss Function**

#Plotting Train Loss vs Validation Loss
plt.plot(history_2.history['loss'])
plt.plot(history_2.history['val_loss'])
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#Plotting Train recall vs Validation recall
plt.plot(history_2.history['recall'])
plt.plot(history_2.history['val_recall'])
plt.title('model recall')
plt.ylabel('recall')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#Predicting the results using best as a threshold
y_train_pred = model_2.predict(X_train)
y_train_pred = (y_train_pred > 0.5)
y_train_pred

#Predicting the results using 0.5 as the threshold.
y_val_pred = model_2.predict(X_val)
y_val_pred = (y_val_pred > 0.5)
y_val_pred

model_name = "NN with Adam & Dropout"

train_metric_df.loc[model_name] = recall_score(y_train,y_train_pred)
valid_metric_df.loc[model_name] = recall_score(y_val,y_val_pred)

# **Classification Report**

#classification report
cr=classification_report(y_train,y_train_pred)
print(cr)

# Checking the model's performance on the validation set
cr = classification_report(y_val, y_val_pred)
print(cr)

# **Confusion Matrix**

#Calculating the confusion matrix
make_confusion_matrix(y_train, y_train_pred)

make_confusion_matrix(y_val, y_val_pred)

# ### Neural Network with Balanced Data (by applying SMOTE) and SGD Optimizer

sm = SMOTE(random_state=42)

# Fit SMOTE on the training data
X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)

print('After UpSampling, the shape of train_X: {}'.format(X_train_smote.shape))
print('After UpSampling, the shape of train_y: {} \n'.format(y_train_smote.shape))

backend.clear_session()
#Fixing the seed for random number generators so that we can ensure we receive the same output everytime
np.random.seed(2)
random.seed(2)
tf.random.set_seed(2)

# Initializing the model
model_3 = Sequential()

# Adding the input layer with 32 neurons and relu as activation function
model_3.add(Dense(32, activation='relu', input_dim=X_train_smote.shape[1]))

# Adding a hidden layer with 16 neurons and relu activation
model_3.add(Dense(16, activation='relu'))

# Adding a second hidden layer with 8 neurons and relu activation
model_3.add(Dense(8, activation='relu'))

# Adding the output layer with 1 neuron and sigmoid activation function (for binary classification)
model_3.add(Dense(1, activation='sigmoid'))

optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

metric = keras.metrics.Recall()

# Compiling the model with binary cross entropy as loss function and recall as the metric
model_3.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[metric])

model_3.summary()

# Fitting the ANN
history_3 = model_3.fit(
    X_train_smote, y_train_smote,
    batch_size=32,
    epochs=100,
    verbose=1,
    validation_data = (X_val, y_val)
)

# **Loss Function**

#Plotting Train Loss vs Validation Loss
plt.plot(history_3.history['loss'])
plt.plot(history_3.history['val_loss'])
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#Plotting Train recall vs Validation recall
plt.plot(history_3.history['recall'])
plt.plot(history_3.history['val_recall'])
plt.title('model recall')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

y_train_pred = model_3.predict(X_train_smote)
#Predicting the results using 0.5 as the threshold
y_train_pred = (y_train_pred > 0.5)
y_train_pred

y_val_pred = model_3.predict(X_val)
#Predicting the results using 0.5 as the threshold
y_val_pred = (y_val_pred > 0.5)
y_val_pred

model_name = "NN with SMOTE & SGD"

train_metric_df.loc[model_name] = recall_score(y_train_smote,y_train_pred)
valid_metric_df.loc[model_name] = recall_score(y_val,y_val_pred)

# **Classification Report**

cr=classification_report(y_train_smote,y_train_pred)
print(cr)

cr = classification_report(y_val, y_val_pred)  ## Checking the model's performance on the validation set
print(cr)

# **Confusion Matrix**

make_confusion_matrix(y_train_smote, y_train_pred)

make_confusion_matrix(y_val, y_val_pred)

# ### Neural Network with Balanced Data (by applying SMOTE) and Adam Optimizer

backend.clear_session()
#Fixing the seed for random number generators so that we can ensure we receive the same output everytime
np.random.seed(2)
random.seed(2)
tf.random.set_seed(2)

# Initializing the model
model_4 = Sequential()

# Adding the input layer with 64 neurons and relu as activation function
model_4.add(Dense(64, activation='relu', input_dim=X_train_smote.shape[1]))

# Adding the first hidden layer with 32 neurons and relu as activation function
model_4.add(Dense(32, activation='relu'))

# Adding the second hidden layer with 16 neurons and relu as activation function
model_4.add(Dense(16, activation='relu'))

# Adding the output layer with 1 neuron and sigmoid activation function (for binary classification)
model_4.add(Dense(1, activation='sigmoid'))

model_4.summary()

optimizer = tf.keras.optimizers.Adam()
metric = keras.metrics.Recall()

# Compiling the model with binary cross entropy as loss function and recall as the metric
model_4.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[metric])

#Fitting the ANN

history_4 = model_4.fit(
    X_train_smote,y_train_smote,
    batch_size=32,
    epochs=20,
    verbose=1,
    validation_data = (X_val,y_val)
)

# **Loss Function**

#Plotting Train Loss vs Validation Loss
plt.plot(history_4.history['loss'])
plt.plot(history_4.history['val_loss'])
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#Plotting Train recall vs Validation recall
plt.plot(history_4.history['recall'])
plt.plot(history_4.history['val_recall'])
plt.title('model recall')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

y_train_pred = model_4.predict(X_train_smote)
#Predicting the results using 0.5 as the threshold
y_train_pred = (y_train_pred > 0.5)
y_train_pred

y_val_pred = model_4.predict(X_val)
#Predicting the results using 0.5 as the threshold
y_val_pred = (y_val_pred > 0.5)
y_val_pred

model_name = "NN with SMOTE & Adam"

train_metric_df.loc[model_name] = recall_score(y_train_smote,y_train_pred)
valid_metric_df.loc[model_name] = recall_score(y_val,y_val_pred)

# **Classification Report**

cr=classification_report(y_train_smote,y_train_pred)
print(cr)

# Using predicted values and true labels for the validation set
cr = classification_report(y_val, model_4.predict(X_val) > 0.5)
print(cr)

# **Confusion Matrix**

make_confusion_matrix(y_train_smote, y_train_pred)

make_confusion_matrix(y_val, model_4.predict(X_val) > 0.5)

# ### Neural Network with Balanced Data (by applying SMOTE), Adam Optimizer, and Dropout

backend.clear_session()
#Fixing the seed for random number generators so that we can ensure we receive the same output everytime
np.random.seed(2)
random.seed(2)
tf.random.set_seed(2)

# Initializing the model
model_5 = Sequential()

# Adding the input layer with 64 neurons and relu as activation function
model_5.add(Dense(64, activation='relu', input_dim=X_train_smote.shape[1]))

# Adding dropout with 0.2 rate to reduce overfitting
model_5.add(Dropout(0.2))

# Adding the first hidden layer with 32 neurons and relu activation function
model_5.add(Dense(32, activation='relu'))

# Adding dropout with 0.2 rate
model_5.add(Dropout(0.2))

# Adding a hidden layer with 8 neurons and relu as activation function
model_5.add(Dense(8, activation='relu'))

# Adding the output layer with 1 neuron and sigmoid activation function for binary classification
model_5.add(Dense(1, activation='sigmoid'))

# Using Adam as the optimizer.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


metric = keras.metrics.Recall()

# Compiling the model with binary cross entropy as loss function and recall as the metric
model_5.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[metric])

model_5.summary()

history_5 = model_5.fit(
    X_train_smote,y_train_smote,
    batch_size=32, ## Specifying the batch size to 32
    epochs=25, ## Specifying the number of epochs as 32
    verbose=1,
    validation_data = (X_val,y_val))

# **Loss Function**

#Plotting Train Loss vs Validation Loss
plt.plot(history_5.history['loss'])
plt.plot(history_5.history['val_loss'])
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#Plotting Train recall vs Validation recall
plt.plot(history_5.history['recall'])
plt.plot(history_5.history['val_recall'])
plt.title('model recall')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

y_train_pred = model_5.predict(X_train_smote)
#Predicting the results using 0.5 as the threshold
y_train_pred = (y_train_pred > 0.5)
y_train_pred

y_val_pred = model_5.predict(X_val)
#Predicting the results using 0.5 as the threshold
y_val_pred = (y_val_pred > 0.5)
y_val_pred

model_name = "NN with SMOTE,Adam & Dropout"

train_metric_df.loc[model_name] = recall_score(y_train_smote,y_train_pred)
valid_metric_df.loc[model_name] = recall_score(y_val,y_val_pred)

# **Classification Report**

cr=classification_report(y_train_smote,y_train_pred)
print(cr)

cr = classification_report(y_val, y_val_pred)
print(cr)

# **Confusion Matrix**

#Calculating the confusion matrix
make_confusion_matrix(y_train_smote, y_train_pred)

make_confusion_matrix(y_val, y_val_pred)

# ## Model Performance Comparison and Final Model Selection

print("Training performance comparison")
train_metric_df

print("Validation set performance comparison")
valid_metric_df

train_metric_df - valid_metric_df

y_test_pred = model_4.predict(X_test)    ## Using the 5th model, with the best recall value - NN with SMOTE and Adam
y_test_pred = (y_test_pred > 0.5)  ## Convert probabilities to binary predictions (0 or 1)
print(y_test_pred)

#lets print classification report
cr=classification_report(y_test,y_test_pred)
print(cr)

#Calculating the confusion matrix
make_confusion_matrix(y_test,y_test_pred)

# ## Actionable Insights and Business Recommendations

# 1. **Focus on Improving Customer Retention:**
# 
#   Given the business context of customer churn (customers leaving the bank), it is crucial for the bank to understand and predict customer churn effectively. With the neural network model, we can predict whether a customer is likely to churn in the next 6 months based on various features and behaviors.
# 
#   Key Model Insights:
# 
#   The NN with SMOTE & Adam model yields the highest recall (0.7331) among all models tested. This model is likely the best for identifying customers at high risk of churning.
#   Recall indicates the model's ability to correctly identify churners (true positives), so a higher recall means fewer churners will be missed. This allows the bank to proactively intervene before customers leave.
# 2. **Prioritize Customer Segments Identified by the Model:**
# 
#   Based on the model's predictions, the bank can prioritize certain customer segments that are at high risk of leaving. For example:
# 
#   Customers who frequently use specific bank services (e.g., loans, savings accounts, or credit cards).
#   Customers with negative experiences in key touchpoints like customer service, mobile app experience, or fees.
#   Long-term customers showing decreasing activity or engagement over time.
#   By identifying these segments early, the bank can offer tailored retention strategies such as:
# 
#   Personalized incentives (e.g., discounts, rewards, or exclusive offers).
#   Dedicated customer service representatives or specialized assistance.
#   Better alignment of products and services to customer preferences.
# 3. **Improve Customer Satisfaction Based on Features Influencing Churn:**
# 
#   The features that drive churn should be examined more closely. Common drivers could be:
# 
#   Poor customer service experiences.
#   High fees or penalties.
#   Complicated account management.
#   Competitor offers: Better rates or customer service from competing banks.
#   Once the model is trained, these features can be further explored to identify key drivers of churn. The bank can then concentrate on improving these service areas:
# 
#   Customer Experience Improvements: Enhance customer service training, increase self-service options (e.g., mobile app features), or streamline onboarding.
#   Fee Restructuring: Assess the competitive landscape and consider revising fees for accounts, loans, or credit cards to be more competitive.
#   Targeted Product Offering: Create personalized packages or offers for high-risk customers to increase loyalty.
# 4. **Model Utilization for Business Strategy:**
# 
#   The NN with SMOTE & Adam model could be used in a real-time scoring system to continuously assess the churn risk of all customers. This can help with:
#   Early Detection: Using the model's predictions to send personalized interventions or offers to customers likely to churn.
#   Campaign Effectiveness: After implementing a customer retention campaign, the model can be used to track its effectiveness by comparing churn predictions before and after the campaign.
# 5. **Implement Continuous Model Monitoring and Improvement:**
# 
#   While the NN with SMOTE & Adam model provides the best recall, continuous monitoring and regular updates to the model are recommended:
# 
#   Regular retraining with updated customer data.
#   Testing new features or changing parameters to improve model accuracy and recall.
#   Customer feedback integration to fine-tune the predictions.
# 6. **Recommendation for Deployment:**
# 
#   Deploy the best-performing model (NN with SMOTE & Adam) into the bank's predictive analytics infrastructure. Use it to:
# 
#   Segment high-risk customers for targeted interventions.
#   Automate churn predictions and incorporate this into the decision-making process of business units (e.g., customer service, sales).
# 
# # Key Takeaways for the Business:
#   
#   **Best Model:** The NN with SMOTE & Adam (recall: 0.7331) is the best performing model for predicting customer churn, offering the highest recall, meaning fewer customers will be missed in the churn prediction process.
#   
#   **Proactive Retention:** By leveraging this model, the bank can take proactive actions to prevent customer churn, including offering incentives, improving customer service, and providing tailored product offerings.
#   
#   **Continuous Improvement:** Regularly updating the model with new data and testing it with additional features can lead to continuous improvements in churn prediction accuracy.
#   
#   **Data-Driven Decisions:** The model allows for data-driven decision-making by pinpointing which customers are at risk and prioritizing their retention.
#   By effectively applying these insights, the bank can reduce customer churn, improve satisfaction, and ultimately enhance customer loyalty and retention.

# <font size=6 color='blue'>Power Ahead</font>
# ___
