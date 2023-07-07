# Classification
# Classification is a fundamental task in machine learning that involves categorizing or assigning data points to predefined classes or categories based on their features
# Classification aims to learn a mapping or decision boundary from labeled training data to make prediciton on new, unseen data

# The K-Nearest Neighbors(KNN)
# The K-Nearest Neighbors(KNN) is a popular algorithm that makes predictions based on the proximity of data points in the feature space
# Training Phase - During the Training Phase, the KNN simply stores the labeled feature vectors of the training data
# Distance Calculation - When a new, unseen data point needs to be classified, the algorithm calculates the distance between that data point and all other data points in the training set
# The most commonly used distance metric is the Euclidean distance 
# Choosing K - The algorithm then selects the K nearest neighbors to the new data point based on the calculated distances 
# Weighted Voting - Once the K nearest neighbors are selected, the algorithm assigns class label to the new data point based on the majority vote among K neighbors
# Prediction - The algorithm returns the predicted class label for the new data point

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
#  'x' and 'y' represent the feature matrix and the target
# test_size = 0.25  specifies that 25% of the data will be allocated for testing while the remaining 75% will be used for training
# random_state = 0 sets the random seed for reproducibility. By using the same seed you ensure that the random splitting of data will be the same each time you run the code
# The train_test_split() function returns four sets of data:
# x_train -> This is the feature matrix for the training set.
# x_test -> This is the feature matrix for the testing set
# y_train -> This is the target vector for the training set
# y_test -> This is the target  vector for the testing set 


# from sklearn.preprocessing import StandardScaler
# This imports the StandardScaler class from the scikit-learn library's preprocessing module
# The 'StandardScaler' class is a commonly used preprocessing technique in machine learning used to standardize or scale the featres of a dataset to have zero mean and unit variance


# sc_x = StandardScaler() creates an instance of the StandardScaler class and assigns it a variable sc_x

# x_train = sc_x.fit_transform(x_train) performs feature scaling or standardization on training data x_train
# .fit_transform() performs two operations:
# .ft() -> this operation computes the mean and standard deviation of features in x_train. It learns the scaling parameters based on the training data
# .transform() -> Applies the scalling transformationto the x_train data using the computed mean and standard deviation. It standardizes the features, ensuring they have zero mean and unit variance
# x_train = sc_x.fit_transform(x_train) The result of the fit_transform() operation is assigned back to x_train. This replaces the original x_train data with scaled version

# x_test = sc_x.fit_transform(x_test) 
# This performs feature scaling or standardization  on the testing data x_test using the StandardScaler



# Fitting the KNN (K-Nearest Neighbors) classifier
# Fitting the KNN (K-Nearest Neighbors) classifier into dataset refers to the process of training or 'fitting' the classifier model using the provided dataset
# It involves teaching the KNN algorithm to learn patterns and relationships in data so that it can make accurate predicitons  or classifications on new unseen data


# classifier = KNeighborsClassifier(n_neighbors = 5, metric = "minkowski", p = 2)
# n_neighbors -> This parameter specifies the number of neighbors to consider when making the prediciton
# metric -> This parameter defines the distance used to measure the similarity or disimilarity between data points . eg Minkowski is a generalization  of the Euclidan and Manhattan  distances
# p -> This parameter is specific to the Minkowski distance calculation  When p=2, it corresponds to the Euclidean distance metric. Setting p=1 would correspond to the Manhattan distance metric


# Confusion or Error Matrix
# This is a useful tool for evaluating the performance of classification algorithms, including KNN, 
# It provides a summary of the classification by displaying the counts of:
# True Positive(TP) -> The number of instances correctly predicted as positive by the model
# True Negative -> The number of instances correctly predicted as negative by the model
# False Positive -> The number of instances incorectly predicted as positive by the model
# Fale Negative -> The number of instances incorrectly predicted as negative by the model
# The confusion matrix helps in understanding the types of errors made by the model and provides insights into its performance

# Example:


# Import the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Read the dataset from a CSV file
dataset = pd.read_csv(r"E:\Data Science Env\ML\user_data.csv")

# Display the dataset
print(dataset)

# Extract the features (Age and EstimatedSalary) and the target variable (Purchased)
x = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, 4].values

# Split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Standardize the features using the StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# Display the standardized training features
# print(x_train)

# Display the target variable of the training set
# print(y_train)

# Create and train the K-Nearest Neighbors classifier
classifier = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
classifier.fit(x_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(x_test)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# Example 2:

# Import the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Read the dataset from a CSV file
dataset = pd.read_csv(r"E:\Data Science Env\ML\user_data.csv")

# Display the dataset
print(dataset)

# Extract the features (Age and EstimatedSalary) and the target variable (Purchased)
x = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, 4].values

# Split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Standardize the features using the StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

# Create a scatter plot of the area and price data points
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='bwr')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')

# Fit a linear regression model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Print the intercept of the linear regression line
print('Intercept:', regressor.intercept_)

# Plotting the linear regression line
x_line = np.linspace(-2, 2, 100).reshape(-1, 1)
y_line = regressor.predict(x_line)
plt.plot(x_line, y_line, color='black')

# Plotting the new predictions
x_test_pred = np.array([[25, 60000], [40, 80000]])  # Example test data
x_test_pred_scaled = sc_x.transform(x_test_pred)
y_test_pred = regressor.predict(x_test_pred_scaled)
plt.scatter(x_test_pred[:, 0], x_test_pred[:, 1], c=y_test_pred, cmap='bwr', marker='X', s=100)

# Display the legend
plt.legend(['Regression Line', 'Predictions'], loc='upper left')

# Show the plot
plt.show()

