# The Logistic Regression Algorithm

# Logisitic Regression is a popular model used for binary classification tasks
# Logistic Regression is primarily a classification algorithm rather than a regression algorithm
# It is based on the concept of the logistic function (i.e sigmoid function) which maps any real-valued input to a value between 0 and 1 
# It is used to transform the output of a linear equation into a probability value


# Implementation Process of the Logistic regression in Python

# from sklearn.metrics import confusion_matrix 
# i.e this Python code imports the confusion_matrix function / class from the sklearn.metrics module
# The classification_matrix is an important tool for evaluating the performance of a classification model
# It allows us to assess the accuracy of the predictions and provides insights into  the types of errors made by the model
# It provides the following information:

# True Positives(TP) -> The number of correctly predicted positive instances
# True Negatives (TN) -> The number of correctly predicted negative instances
# False Positives (FP) -> The number of instances predicted as positive, but they are actually negative
# False Negatives (FN) -> The number of instances predicted as negative, but they are actually positive


# x = dataset.iloc[: , [2, 4]] -> This used to extract the features from the dataset that will be used as input variables for a machine learning model
# [: , [2, 4]] -> This part of the code selects all rows, i.e ( " : ") and specific columns i.e (" [2, 4] ")

# y = dataset.iloc[:, 4] -> (":") is used to extract all rows, ("4") is used to extract the column 4


# from sklearn.model_selection import train_test_split -> this is important for splitting the dataset into training and testing sets
# model_selection -> this is a submodule within scikit-learn that includes various functions and classes for model selection and evaluation
# train_test_split -> It is a function within the model_selection that allows the user to split the dataset and training set
# it takes the input features and the target variables and splits them into four sets:
# x_train and y_train, which represents the training data
# x_test and y_test which represent the training data


# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state =  0)
# i.e x and y represent the input features and target variables repsectively
# test_size -> is a parameter that determines the ratio or proportion of dataset that will be allocated for the testing set. 
# train_size = 0.25 ->indicates that 25% of the data will be used for testing while the remaining 75% will be used for training   
# random_state  -> This is an optional parameter that sets the random seed for reproducibility  


# from sklearn.preprocessing import StandardScaler
# i.e is important for performing scaling on the dataset
# StandardScaler -> This is a class from the preprocessing module of scikit-learn that provides a method for standaedizing features by removing the mean  and scalling to unit variance     


# sc_x = StandardScaler -> is important for creating an instance of the StandarScaler class from scikit-learn preprocessing module 

# x_train = sc_x.fit_transform (x_train)
# Feature Scalling  is a crucial preprocesing step in ML. It ensures that all features have similar scales and ranges which can improve the performance and convergence of many machine learning algorithms 
# fit_transform() is a convinient method provided by the StandardScaler onject
# It performs twp operations:
# Fitting the scaler on the training data to learn the scaling parameters and then transforming the training data based on those parameters


# from sklearn.linear_model import LogisticRegression
# This is a Python code used to import the LogisticRegression class from the scikit-library


# classifier = LogisticRegression(random_state = 0)
# This creates an instance of the LogisticRegression class and assigns it to the variable 'classifier'
# random_state = 0 -> This parameter is used to seed the random number generator, ensuring that the results are reproducible. By setting it equal to 0, we obtain consistent results each time we run the code


# classifier.fit(x_train, y_train)
# This step trains logistic model using the training data
# .fit() estimates the coefficients or weights of the logisitc regression model based on the provided training data, optimizes them iteratively and prepares the model  for making predicitons on the new data



# y_pred = classifier.predict(x_test)
# This step uses the trained logistic model to make new predicitons on new unseen data
# the predict() methods takes the input features, x_test and generates predicted class labels y_pred based on the learned decision boundary

# cm = confusion_matrix(y_pred, y_test)
# This steps summarizes the performance of a classification model by comparing the predicted class labels ("y_pred") with the actual classes ("y_test")



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

# Read the dataset from a CSV file
dataset = pd.read_csv(r"E:\Data Science Env\ML\user_data.csv")

# Display the dataset
dataset

# Extract the features from columns 2 and 4
x = dataset.iloc[:, [2, 4]].values
print(x)

# Extract the target variable from column 4
y = dataset.iloc[:, 4].values
print(y)

from sklearn.model_selection import train_test_split

# Split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler

# Standardize the training and test features using StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

# Display the standardized features
x
x_test

from sklearn.linear_model import LogisticRegression

# Create a Logistic Regression classifier object
classifier = LogisticRegression(random_state=0)

# Fit the classifier to the training data
classifier.fit(x_train, y_train)

# Predict the target variable for the test data
y_pred = classifier.predict(x_test)
print(y_pred)

import seaborn as sns

# Create a confusion matrix
cm = confusion_matrix(y_pred, y_test)
print(cm)

# Visualize the confusion matrix using a heatmap
sns.heatmap(cm)
