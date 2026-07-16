"""
Safety Helmet Detection — CNN Image Classification

Binary image classifier detecting whether workers are wearing safety
helmets, comparing a custom CNN against VGG-16 transfer learning
approaches (base, +FFNN, +data augmentation).
"""

# # Safety Helmet Detection — CNN Image Classification
# 
# *Binary image classifier detecting whether workers are wearing safety helmets, comparing a custom CNN against VGG-16 transfer learning approaches.*
# 
# # **Problem Statement**

# ## **Business Context**

# Workplace safety in hazardous environments like construction sites and industrial plants depends heavily on compliance with safety equipment rules — helmets in particular protect against head injuries from falling objects and machinery. Manual monitoring of helmet compliance is error-prone and doesn't scale well across large operations.
# 
# This project builds an automated image classification system to detect whether workers are wearing safety helmets, aiming to improve safety enforcement accuracy and scale monitoring beyond what manual oversight can achieve.

# ## **Objective**

# As a data scientist at SafeGuard Corp, you are tasked with developing an image classification model that classifies images into one of two categories:
# - **With Helmet:** Workers wearing safety helmets.
# - **Without Helmet:** Workers not wearing safety helmets.

# ## **Data Description**

# The dataset consists of **631 images**, equally divided into two categories:
# 
# - **With Helmet:** 311 images showing workers wearing helmets.
# - **Without Helmet:** 320 images showing workers not wearing helmets.
# 
# **Dataset Characteristics:**
# - **Variations in Conditions:** Images include diverse environments such as construction sites, factories, and industrial settings, with variations in lighting, angles, and worker postures to simulate real-world conditions.
# - **Worker Activities:** Workers are depicted in different actions such as standing, using tools, or moving, ensuring robust model learning for various scenarios.

# # **Installing and Importing the Necessary Libraries**

# Install dependencies (run once):
# pip install tensorflow[and-cuda] numpy==1.25.2 -q

import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print(tf.__version__)

# **Note:**
# 
# - After running the above cell, kindly restart the notebook kernel (for Jupyter Notebook) or runtime (for Google Colab) and run all cells sequentially from the next cell.
# 
# - On executing the above line of code, you might see a warning regarding package dependencies. This error message can be ignored as the above code ensures that all necessary libraries and their dependencies are maintained to successfully execute the code in this notebook.

import os
import random
import numpy as np                                                                               # Importing numpy for Matrix Operations
import pandas as pd
import seaborn as sns
import matplotlib.image as mpimg                                                                              # Importing pandas to read CSV files
import matplotlib.pyplot as plt                                                                  # Importting matplotlib for Plotting and visualizing images
import math                                                                                      # Importing math module to perform mathematical operations
import cv2


# Tensorflow modules
import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator                              # Importing the ImageDataGenerator for data augmentation
from tensorflow.keras.models import Sequential                                                   # Importing the sequential module to define a sequential model
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,BatchNormalization # Defining all the layers to build our CNN Model
from tensorflow.keras.optimizers import Adam,SGD                                                 # Importing the optimizers which can be used in our model
from sklearn import preprocessing                                                                # Importing the preprocessing module to preprocess the data
from sklearn.model_selection import train_test_split                                             # Importing train_test_split function to split the data into train and test
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Model
from keras.applications.vgg16 import VGG16                                               # Importing confusion_matrix to plot the confusion matrix

# Display images using OpenCV
# cv2_imshow is Colab-specific; outside Colab use cv2.imshow() or plt.imshow() instead
try:
    from google.colab.patches import cv2_imshow
except ImportError:
    def cv2_imshow(img):
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.show()

#Imports functions for evaluating the performance of machine learning models
from sklearn.metrics import confusion_matrix, f1_score,accuracy_score, recall_score, precision_score, classification_report
from sklearn.metrics import mean_squared_error as mse                                                 # Importing cv2_imshow from google.patches to display images

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Set the seed using keras.utils.set_random_seed. This will set:
# 1) `numpy` seed
# 2) backend random seed
# 3) `python` random seed
tf.keras.utils.set_random_seed(812)

# # **Data Overview**

# ##Loading the data

import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math
import cv2


# Tensorflow modules
import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,BatchNormalization
from tensorflow.keras.optimizers import Adam,SGD
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Model
from keras.applications.vgg16 import VGG16

# Display images using OpenCV
# cv2_imshow is Colab-specific; outside Colab use cv2.imshow() or plt.imshow() instead
try:
    from google.colab.patches import cv2_imshow
except ImportError:
    def cv2_imshow(img):
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.show()

#Imports functions for evaluating the performance of machine learning models
from sklearn.metrics import confusion_matrix, f1_score,accuracy_score, recall_score, precision_score, classification_report
from sklearn.metrics import mean_squared_error as mse

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Set the seed using keras.utils.set_random_seed. This will set:
# 1) `numpy` seed
# 2) backend random seed
# 3) `python` random seed
tf.keras.utils.set_random_seed(812)

# Colab-specific data mount, not needed outside Colab:
# from google.colab import drive
# drive.mount('/content/drive')

images = np.load('images_proj.npy').reshape(631, 200, 200, 3)

labels = pd.read_csv('Labels_proj.csv')

print(images.shape)
print(labels.shape)

# # **Exploratory Data Analysis**

# ###Plot random images from each of the classes and print their corresponding labels.

helmet_indices = np.where(labels == 1)[0]
no_helmet_indices = np.where(labels == 0)[1]

# Select one image from each class
helmet_img = images[np.random.choice(helmet_indices)]
no_helmet_img = images[np.random.choice(no_helmet_indices)]

# Plot the images
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# Display "With Helmet" image
axes[0].imshow(helmet_img)
axes[0].set_title("Worker WITH Helmet")
axes[0].axis('off')

# Display "Without Helmet" image
axes[1].imshow(no_helmet_img)
axes[1].set_title("Worker WITHOUT Helmet")
axes[1].axis('off')

# Show the plots
plt.tight_layout()
plt.show()

# ## Checking for class imbalance

# Create a count plot
plt.figure(figsize=(6, 4))
ax = sns.countplot(x=labels.iloc[:, 0], palette=['red', 'green'])

# Add exact counts on top of bars
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=10, )

# Add labels
plt.xlabel("Class Labels", fontsize=12)
plt.ylabel("Number of Images", fontsize=12)
plt.title("Count of Images per Class", fontsize=14)
plt.xticks(ticks=[0, 1], labels=["Without Helmet (0)", "With Helmet (1)"])  # Rename x-axis labels

# Show plot
plt.show()

# # **Data Preprocessing**

# ## Converting images to grayscale

images_gray = []
for i in range(len(images)):
    img_gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    images_gray.append(img_gray)

# Display a sample grayscale image
n = 0
cv2_imshow(images_gray[n])

# ### Splitting the dataset

from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(np.array(images),labels , test_size=0.3, random_state=42,stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(X_temp,y_temp , test_size=0.5, random_state=42,stratify=y_temp)

print(X_train.shape,y_train.shape)
print(X_val.shape,y_val.shape)
print(X_test.shape,y_test.shape)

# ### Data Normalization

X_train_normalized = X_train.astype('float32') / 255.0
X_val_normalized = X_val.astype('float32') / 255.0
X_test_normalized = X_test.astype('float32') / 255.0

# # **Model Building**

# ##Model Evaluation Criterion

# ## Utility Functions

# defining a function to compute different metrics to check performance of a classification model built using statsmodels
def model_performance_classification(model, predictors, target):
    """
    Function to compute different metrics to check classification model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    """

    # checking which probabilities are greater than threshold
    pred = model.predict(predictors).reshape(-1)>0.5

    target = target.to_numpy().reshape(-1)


    acc = accuracy_score(target, pred)  # to compute Accuracy
    recall = recall_score(target, pred, average='weighted')  # to compute Recall
    precision = precision_score(target, pred, average='weighted')  # to compute Precision
    f1 = f1_score(target, pred, average='weighted')  # to compute F1-score

    # creating a dataframe of metrics
    df_perf = pd.DataFrame({"Accuracy": acc, "Recall": recall, "Precision": precision, "F1 Score": f1,},index=[0],)

    return df_perf

def plot_confusion_matrix(model,predictors,target,ml=False):
    """
    Function to plot the confusion matrix

    model: classifier
    predictors: independent variables
    target: dependent variable
    ml: To specify if the model used is an sklearn ML model or not (True means ML model)
    """

    # checking which probabilities are greater than threshold
    pred = model.predict(predictors).reshape(-1)>0.5

    target = target.to_numpy().reshape(-1)

    # Plotting the Confusion Matrix using confusion matrix() function which is also predefined tensorflow module
    confusion_matrix = tf.math.confusion_matrix(target,pred)
    f, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        linewidths=.4,
        fmt="d",
        square=True,
        ax=ax
    )
    plt.show()

# ##Model 1: Simple Convolutional Neural Network (CNN)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall # Import Recall metric


# Initializing Model
model_1 = Sequential()

# Convolutional layers
model_1.add(Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(200, 200, 3)))
model_1.add(MaxPooling2D((4, 4), padding='same'))
model_1.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
model_1.add(MaxPooling2D((2, 2), padding='same'))
model_1.add(Conv2D(128, (3, 3), activation='relu', padding="same"))

# Flatten and Dense layers
model_1.add(Flatten())
model_1.add(Dense(4, activation='relu'))
model_1.add(Dense(1, activation='sigmoid'))  # Binary classification

# Compile with Adam Optimizer
opt = Adam(learning_rate=0.001)
model_1.compile(optimizer=opt, loss='binary_crossentropy', metrics=["accuracy", Recall()])

# Summary
model_1.summary()

history_1 = model_1.fit(
            X_train_normalized, y_train,
            epochs=15,
            validation_data=(X_val_normalized,y_val),
            shuffle=True,
            batch_size=32,
            verbose=2
)

plt.plot(history_1.history['accuracy'])              # Train accuracy
plt.plot(history_1.history['val_accuracy'])          # Validation accuracy
plt.title('Model Accuracy')                          # Plot title
plt.ylabel('Accuracy')                               # Y-axis label
plt.xlabel('Epoch')                                  # X-axis label
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

model_1_train_perf = model_performance_classification(model_1, X_train_normalized,y_train)

print("Train performance metrics")
print(model_1_train_perf)

plot_confusion_matrix(model_1,X_train_normalized,y_train)

model_1_valid_perf = model_performance_classification(model_1, X_val_normalized,y_val)

print("Validation performance metrics")
print(model_1_valid_perf)

plot_confusion_matrix(model_1,X_val_normalized,y_val)

# ### Vizualizing the predictions

# For index 2
plt.figure(figsize=(2,2))
plt.imshow(X_val[12])
plt.show()
prediction = model_1.predict(X_val_normalized[2].reshape(1,200,200,3))
predicted_label = prediction[0][0]>0.5  # Extract the predicted class label
print('Predicted Label:', 1 if predicted_label else 0)
# Fix indexing issue in y_val
true_label = y_val.iloc[12]
print('True Label:', true_label)

# For index 33
plt.figure(figsize=(2,2))
plt.imshow(X_val[33])
plt.show()
prediction = model_1.predict(X_val_normalized[2].reshape(1,200,200,3))
predicted_label = prediction[0][0]>0.5  # Extract the predicted class label
print('Predicted Label:', 1 if predicted_label else 0)
# Fix indexing issue in y_val
true_label = y_val.iloc[33]
print('True Label:', true_label)

# ## Model 2: (VGG-16 (Base))

vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
vgg_model.summary()

# Making all the layers of the VGG model non-trainable. i.e. freezing them
for layer in vgg_model.layers:
    layer.trainable = False

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

model_2 = Sequential()

# Adding the convolutional part of the VGG16 model from above
model_2.add(vgg_model)

# Flattening the output of the VGG16 model because it is from a convolutional layer
model_2.add(Flatten())

# Adding a dense output layer
model_2.add(Dense(1, activation='sigmoid'))  # Binary classification

opt = Adam(learning_rate=0.0001)

# Compile model
model_2.compile(optimizer=opt,
                loss=keras.losses.BinaryCrossentropy(),
                metrics=["accuracy"])

# Generating the summary of the model
model_2.summary()

train_datagen = ImageDataGenerator()

# Epochs
epochs = 15  # Typical starting point for training

# Batch size
batch_size = 32  # Common batch size for image models

history_2 = model_2.fit(train_datagen.flow(X_train_normalized, y_train,
                                           batch_size=batch_size,
                                           seed=42,
                                           shuffle=False),
                        epochs=epochs,
                        steps_per_epoch=X_train_normalized.shape[0] // batch_size,
                        validation_data=(X_val_normalized, y_val),
                        verbose=1)

plt.plot(history_2.history['loss'])                   # Train loss
plt.plot(history_2.history['val_loss'])               # Validation loss
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

model_2_train_perf = model_performance_classification(model_2,X_train_normalized,y_train)

print("Train performance metrics")
print(model_2_train_perf)

plot_confusion_matrix(model=model_2,predictors=X_train_normalized,target=y_train)

model_2_valid_perf = model_performance_classification(model_2, X_val_normalized,y_val)

print("Validation performance metrics")
print(model_2_valid_perf)

plot_confusion_matrix(model_2,X_val_normalized,y_val)

# ### Visualizing the prediction:

# First sample
i = 0
plt.figure(figsize=(2, 2))
plt.imshow(X_val[i])
plt.title("Sample Image {}".format(i))
plt.axis('off')
plt.show()

prediction = model_2.predict(X_val_normalized[i].reshape(1, 200, 200, 3))
predicted_label = int(prediction[0][0] > 0.5)
print('Predicted Label:', predicted_label)

true_label = y_val.iloc[i] if hasattr(y_val, 'iloc') else y_val[i]
print('True Label:', true_label)


# Second sample
j = 1
plt.figure(figsize=(2, 2))
plt.imshow(X_val[j])
plt.title("Sample Image {}".format(j))
plt.axis('off')
plt.show()

prediction = model_2.predict(X_val_normalized[j].reshape(1, 200, 200, 3))
predicted_label = int(prediction[0][0] > 0.5)
print('Predicted Label:', predicted_label)

true_label = y_val.iloc[j] if hasattr(y_val, 'iloc') else y_val[j]
print('True Label:', true_label)

# ## Model 3: (VGG-16 (Base + FFNN))

model_3 = Sequential()

# Adding the convolutional part of the VGG16 model from above
model_3.add(vgg_model)

# Flattening the output of the VGG16 model because it is from a convolutional layer
model_3.add(Flatten())

# Adding the Feed Forward neural network
model_3.add(Dense(256, activation='relu'))   # First dense layer
model_3.add(Dropout(rate=0.5))               # Dropout to prevent overfitting
model_3.add(Dense(128, activation='relu'))   # Second dense layer

# Adding a dense output layer (binary classification)
model_3.add(Dense(1, activation='sigmoid'))  # Output neuron with sigmoid for binary classification

opt = Adam(learning_rate=1e-4)

model_3.compile(optimizer=opt, loss=keras.losses.BinaryCrossentropy(), metrics=["accuracy"])

model_3.summary()

history_3 = model_3.fit(train_datagen.flow(X_train_normalized, y_train,
                                           batch_size=32,  # Common value
                                           seed=42,
                                           shuffle=False),
                        epochs=15,  # Start with 15; adjust based on validation performance
                        steps_per_epoch=X_train_normalized.shape[0] // 32,
                        validation_data=(X_val_normalized, y_val),
                        verbose=1)

plt.plot(history_2.history['loss'])
plt.plot(history_2.history['val_loss'])
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

model_3_train_perf = model_performance_classification(model_3, X_train_normalized,y_train)

print("Train performance metrics")
print(model_3_train_perf)

plot_confusion_matrix(model_3,X_train_normalized,y_train)

model_3_valid_perf = model_performance_classification(model_3, X_val_normalized,y_val)

print("Validation performance metrics")
print(model_3_valid_perf)

plot_confusion_matrix(model_3,X_val_normalized,y_val)

# #### Visualizing the predictions

# First example
i = 5
plt.figure(figsize=(2, 2))
plt.imshow(X_val[i])  # Display original image
plt.title(f'Validation Sample {i}')
plt.axis('off')
plt.show()

prediction = model_2.predict(X_val_normalized[i].reshape(1, 200, 200, 3))  # Predict
predicted_label = 1 if prediction[0][0] > 0.5 else 0
print('Predicted Label:', predicted_label)

true_label = y_val.iloc[i] if hasattr(y_val, 'iloc') else y_val[i]
print('True Label:', true_label)

# Second example
j = 1
plt.figure(figsize=(2, 2))
plt.imshow(X_val[j])
plt.title(f'Validation Sample {j}')
plt.axis('off')
plt.show()

prediction = model_2.predict(X_val_normalized[j].reshape(1, 200, 200, 3))
predicted_label = 1 if prediction[0][0] > 0.5 else 0
print('Predicted Label:', predicted_label)

true_label = y_val.iloc[j] if hasattr(y_val, 'iloc') else y_val[j]
print('True Label:', true_label)

# ## Model 4: (VGG-16 (Base + FFNN + Data Augmentation)

# - In most of the real-world case studies, it is challenging to acquire a large number of images and then train CNNs.
# - To overcome this problem, one approach we might consider is **Data Augmentation**.
# - CNNs have the property of **translational invariance**, which means they can recognise an object even if its appearance shifts translationally in some way. - Taking this attribute into account, we can augment the images using the techniques listed below
# 
#     -  Horizontal Flip (should be set to True/False)
#     -  Vertical Flip (should be set to True/False)
#     -  Height Shift (should be between 0 and 1)
#     -  Width Shift (should be between 0 and 1)
#     -  Rotation (should be between 0 and 180)
#     -  Shear (should be between 0 and 1)
#     -  Zoom (should be between 0 and 1) etc.
# 
# Remember, **data augmentation should not be used in the validation/test data set**.

model_4 = Sequential()

# Adding the convolutional part of the VGG16 model from above
model_4.add(vgg_model)

# Flattening the output of the VGG16 model because it is from a convolutional layer
model_4.add(Flatten())

# Adding the Feed Forward neural network
model_4.add(Dense(256, activation='relu'))    # First dense layer
model_4.add(Dropout(rate=0.5))                # Dropout to reduce overfitting
model_4.add(Dense(128, activation='relu'))    # Second dense layer

# Adding a dense output layer for binary classification
model_4.add(Dense(1, activation='sigmoid'))   # 1 neuron + sigmoid for binary output

opt = Adam(learning_rate=0.0001)  # Typical learning rate for fine-tuning

# Compile model
model_4.compile(optimizer=opt,
                loss=keras.losses.BinaryCrossentropy(),
                metrics=["accuracy"])  # Common metric for binary classification

model_4.summary()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Applying data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,          # Rotate images up to 20 degrees
    width_shift_range=0.2,      # Shift image horizontally by 20%
    height_shift_range=0.2,     # Shift image vertically by 20%
    shear_range=0.15,           # Shear intensity (shear angle in counter-clockwise direction)
    zoom_range=0.2,             # Zoom in/out by up to 20%
    fill_mode='nearest'         # Fill pixels with nearest filled values
)

batch_size = 32  # You can also try 16 or 64 depending on GPU/CPU memory

history_4 = model_4.fit(
    train_datagen.flow(X_train_normalized, y_train,
                       batch_size=batch_size,
                       seed=42,
                       shuffle=False),
    epochs=epochs,
    steps_per_epoch=X_train_normalized.shape[0] // batch_size,
    validation_data=(X_val_normalized, y_val),
    verbose=1
)

plt.plot(history_2.history['loss'])                 # Train loss
plt.plot(history_2.history['val_loss'])             # Validation loss
plt.title('Model Loss')                              # Plot title
plt.ylabel('Loss')                                   # Y-axis label
plt.xlabel('Epoch')                                  # X-axis label
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

model_4_train_perf = model_performance_classification(model_4, X_train_normalized,y_train)

print("Train performance metrics")
print(model_4_train_perf)

plot_confusion_matrix(model_4,X_train_normalized,y_train)

model_4_valid_perf = model_performance_classification(model_4, X_val_normalized,y_val)

print("Validation performance metrics")
print(model_4_valid_perf)

plot_confusion_matrix(model_4,X_val_normalized,y_val)

# #### Visualizing the predictions

i = 10  # Example index, change as needed

plt.figure(figsize=(2,2))
plt.imshow(X_val[i])
plt.show()
prediction = model_2.predict(X_val_normalized[i].reshape(1, 200, 200, 3))
predicted_label = prediction[0][0] > 0.5
print('Predicted Label:', 1 if predicted_label else 0)
true_label = y_val.iloc[i]
print('True Label:', true_label)

plt.figure(figsize=(2,2))
plt.imshow(X_val[9])
plt.show()
prediction = model_2.predict(X_val_normalized[i].reshape(1, 200, 200, 3))
predicted_label = prediction[0][0] > 0.5
print('Predicted Label:', 1 if predicted_label else 0)
true_label = y_val.iloc[i]
print('True Label:', true_label)

# # **Model Performance Comparison and Final Model Selection**

# training performance comparison

models_train_comp_df = pd.concat(
    [
        model_1_train_perf.T,
        model_2_train_perf.T,
        model_3_train_perf.T,
        model_4_train_perf.T,
    ],
    axis=1,
)
models_train_comp_df.columns = [
    "Simple Convolutional Neural Network (CNN)","VGG-16 (Base)","VGG-16 (Base+FFNN)","VGG-16 (Base+FFNN+Data Aug)"
]

models_valid_comp_df = pd.concat(
    [
        model_1_valid_perf.T,
        model_2_valid_perf.T,
        model_3_valid_perf.T,
        model_4_valid_perf.T

    ],
    axis=1,
)
models_valid_comp_df.columns = [
 "Simple Convolutional Neural Network (CNN)","VGG-16 (Base)","VGG-16 (Base+FFNN)","VGG-16 (Base+FFNN+Data Aug)"
]

models_train_comp_df

models_valid_comp_df

models_train_comp_df - models_valid_comp_df

# ## Test Performance

model_test_perf = model_performance_classification(model_3, X_test_normalized, y_test)

model_test_perf

plot_confusion_matrix(model_3, X_test_normalized, y_test)

# # **Actionable Insights & Recommendations**

# # **Actionable Insights**
# **1. Consistent Model Performance Across Training and Testing**
# 
# All models, especially the VGG-16-based variants, demonstrated high accuracy and consistency across both training and test datasets.
# Metrics such as precision, recall, and F1 score remain stable, indicating the models are likely generalizing well within the current dataset scope.
# 
# **2. Limited Impact of Data Augmentation on Accuracy**
# 
# Adding data augmentation did not alter the evaluation metrics on the test set, suggesting that the original dataset already includes sufficient variability (e.g., lighting, posture, background).
# 
# **3. Pretrained Models Show Advantage Over Basic CNN**
# 
# VGG-16-based models outperformed the basic CNN in terms of test performance, indicating that leveraging pretrained feature extractors is beneficial, especially with moderate-sized datasets.
# 
# **4. Low Indications of Overfitting**
# 
# The minimal difference between training and test scores suggests that the models are not overfitting on the training data, even when achieving high metrics.
# 
# # **Recommendations**
# 
# **1. Proceed with the VGG-16 + FFNN + Data Augmentation Model**
# 
# This configuration provides consistent evaluation metrics across varied conditions and is suitable for deployment in real-world environments.
# Prepare the model for deployment using formats such as TensorFlow Lite, ONNX, or via API endpoints depending on infrastructure.
# 
# **2. Deploy on Edge Devices for On-Site Monitoring**
# 
# For construction and industrial sites, consider deploying the model on edge hardware (e.g., Nvidia Jetson, Coral TPU) to enable real-time detection with reduced latency.
# 
# **3. Integrate with Operational Safety Systems**
# 
# Embed the detection model into existing video surveillance systems or attendance logs to flag instances of non-compliance.
# Alerts can be configured for operational oversight or worker safety audits.
# 
# **4. Implement Human Feedback and Periodic Evaluation**
# 
# Establish a feedback mechanism where flagged images can be reviewed manually. Use these inputs to further train or validate the model over time.
# Monitor model outputs periodically to detect any drop in performance due to changing work environments or new camera installations.
# 
# **5. Plan for Dataset Expansion and Annotation**
# 
# To maintain performance as deployment scales, collect and incorporate new images from different work environments and edge cases (e.g., partial helmet visibility, multiple workers).
# This will enhance the model’s adaptability to diverse scenarios.
# 
# **6. Monitor for Model Drift Post-Deployment**
# 
# Introduce a routine evaluation schedule (e.g., quarterly) using recent operational data to assess whether the model continues to perform reliably.
# 
# **7. Explore Future Capabilities**
# 
# Consider extending the solution to multi-class or multi-label classification, detecting other personal protective equipment (PPE) such as safety vests or gloves.
# Object detection frameworks (e.g., YOLO, SSD) can be explored for this enhancement.

# <font size=5 color='blue'>Power Ahead!</font>
# ___
