# Regression

# Regression is a statistical modeling technique used to establish the relationship between a dependent variable and one or more independent variables. 
# It aims to predict or estimate the value of the dependent variable based on the values of the independent variables.

# Types of Regression

# 1. Linear Regression: 
# Linear regression is a basic and widely used regression technique. 
# It assumes a linear relationship between the dependent variable and the independent variable(s). 
# The goal is to fit a line that best represents the relationship between the variables. 
# Linear regression can be further classified into simple linear regression and multiple linear regression

# a. Simple Linear Regression: 
# In simple linear regression, there is only one independent variable used to predict the dependent variable. 
# It involves fitting a straight line to the data points to represent the relationship between the variables. 
# The equation of a simple linear regression model is typically represented as: Y = β0 + β1X + ε,
# where Y is the dependent variable, X is the independent variable, β0 is the intercept, β1 is the coefficient (slope), and ε is the error term

# b. Multiple Linear Regression: 
# Multiple linear regression involves more than one independent variable to predict the dependent variable. 
# It extends the concept of simple linear regression to multiple dimensions. 
# The equation of a multiple linear regression model is: Y = β0 + β1X1 + β2X2 + ... + βnXn + ε,
# where Y is the dependent variable, X1, X2, ..., Xn are the independent variables, β0 is the intercept, β1, β2, ..., βn are the coefficients, and ε is the error term

# 2. Nonlinear Regression: 
# Nonlinear regression is used when the relationship between the dependent and independent variables is not linear. 
# It allows for more complex relationships, such as curves or other nonlinear shapes. 
# Nonlinear regression models can take various forms, and the choice of the model depends on the specific problem and data. 
# The equation of a nonlinear regression model is typically represented as a function of the independent variables, and the parameters are estimated using optimization techniques


# Simple Linear Regression in Python Example:
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Read the CSV file containing home prices into a DataFrame
df = pd.read_csv('E:\Data Science Env\ML\homeprices.csv')

# Set the plot style to dark
plt.style.use('dark_background')

# Set the x-axis label for the scatter plot
plt.xlabel('Area(sqr ft)')

# Set the y-axis label for the scatter plot
plt.ylabel('Price(US$)')

# Create a scatter plot of the area and price data points
plt.scatter(df['Area'], df['Price'], color='yellow', marker='+')

# Create an instance of the LinearRegression class
reg = LinearRegression()

# Fit the linear regression model using the area as the independent variable and price as the dependent variable
reg.fit(df[['Area']], df['Price'])

# Predict the price for a new area value of 3300
prediction = reg.predict([[3300]])

# Print the coefficient(slope) of the linear regression line
print('Coefficient:', reg.coef_[0])

# Print the intercept of the linear regression line
print('Intercept:', reg.intercept_)

# Generate predicted values for the entire range of the dataset
predicted_values = reg.predict(df[['Area']])

# Plotting the linear regression line
plt.plot(df['Area'], predicted_values, color='blue', label='Linear Regression')

# Plotting the new predictions
plt.scatter([3300], prediction, color='red', marker='x', label='New Prediction')

# Display the legend
plt.legend()

# Show the plot
plt.show()

# Ex2:

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data
df = pd.DataFrame({
    'Hours': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Score': [65, 78, 83, 88, 90, 92, 95, 97, 99]
})

# set the plot style to dark
plt.style.use('dark_background')

# plotting the data points
plt.scatter(df['Hours'], df['Score'], marker='.', color='yellow')
plt.xlabel('Hours')
plt.ylabel('Score')
plt.title('Hours vs. Score')

# Creating an instance of linear regression
reg = LinearRegression()

# Fitting the linear regression using hours as the independent variable and score as the depedent variable
reg.fit(df[['Hours']], df['Score'])

# Predicting the score for a new number of hours, (12 in this example)
prediction = reg.predict([[12]])

# Printing the coefficient(slope) of the linear regression line
print('Coefficient: ', reg.coef_[0])

# Printing the intercept of the linear regression line
print("Intercept: ", reg.intercept_)

# Plotting the linear regression line
plt.plot(df['Hours'], reg.predict(df[['Hours']]), color= 'blue', label= 'Linear Regression')

# Adding the predicted point to the plot
plt.scatter([12], prediction, color = 'red', marker= 'x', label='Prediciton')

plt.legend()

plt.show()

# 2. Modelling the Multiple Linear Regression

# Accessing Data / Manipulating Data in Python

# .iloc[:, :-1] -> The ":" before the comma indicates we want to select all rows, and the ":" after the comma indicates we want to select all the columns except the last column

# .iloc[:, -1] -> The ":" before the comma indicates we want to select all rows, and the "-1" after the comma indicates we want only select the last column

# .iloc[2:5, :] -> This selects rows 3, 4, 5 in the original DataFrame and all columns

# .iloc[[1,3,5], [2,4]] -> This selects rows 2, 4, and 6 in the original DataFrame and Columns 3 and 5 in the original DataFrame

# Dealing with Categorical Data in a dataset
# One common approach to deal with Category Data is to encode them into numerical values that can be used for input for machine learning algorithms
# We can use the OneHotEncoding:

# Import the OneHotEncoder library from scikit learn i.e from sklearn.preprocessing import  OneHotEncoder
# Load and Preprocess Data 
# Identify Categorical Columns
# Create a ColumnTransformer i.e from sklearn import ColumnTransformer  
# Specify the Transformer i.e ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3]), remainder = 'passthrough'])
# i.e 'transformers' parameter specifies the transformations to be applied to the columns of the dataset. In this case, a tupple is provided with 3 elements, transformer name as "encoder", the OneHotEncoder() as instance, and the column index '[3]'
# The remainder parameter is set to 'passthrough', as this ensures that any remaining columns are not transformed
# Apply the transformation, ColumnTransformer is applied to the input data 'x' using the 'fit_transform()' method i.e x=np.array(ct.fit_transform(x))


# Splitting the DataSet into training Sets
# The purpose of Splitting the data set into training sets is to evaluate the perfomance of the model on unseen data and access its generalization ability
# The general purpose is dividing the dataset into two seperate subsets, one for training the model and the other for evaluating its performance 
# Dataset - The Dataset consists of input features(x) and the corresponding target values (y)   
# Training Set - The training Set is a subset of the dataset used to train the machine learning model   
# Test Set - The test set is a seperate subset of the dataset that is not used during the model training phase, it simulates real world, unseen data that the model will encounter during deployment
# Splitting Strategy - Some of the common splitting startegies include a  70% -80% of the data being used for training while the remaining 20%-30% being reserved for testing
# i.e from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


# Training Model on the training set
# This involves fitting machine learning algorithm to the training data to learn patterns and relationships between the input features and the corresponding target variable
# Training Set - contains input features(x_train) and the corresponding target variable(y_train)
# Model Initialization - This is choosing the appropriate algorithm or model based on the problem you are trying to solve
# Training Process - Involves providing the training set to the model and adjusting its internal parameters based on the input features and target variable
# Model Learning -
# Objective Function -
# Iterations and Epochs -
# Model Outputs - 
# i.e   from sklearn.linear_model import Linear Regression -> Importing recquired Libraries 
# i.e regressor = LinearRegression() -> Instatiating the Regressor
# i.e regressor.fit(x_train, y_train)  -> Fitting the Model


# Predicting the test results 
# After training the model, the model is then used to make predicitons on the test set, which is the unseen data 
# i.e y_pred = regressor.predict(x_test)
# i.e df = pd.DataFrame({'Real Values' : y_test, 'Predicted Values' : y_pred}



# Measuring Accuracy and Evaluating the Performance of a Regression model
# It involves assessing how well the model's prediction align with true target values
# Some of the Common Evaluation Metrics include:
# Mean Squared Error ->  This metric calculates the average squared difference between the predicted and true values it penalizes larger prediciton errors more heavily
# Root Mean Squared Error -> RMSE is the square root of the MSE and provides a measure of of the average magnitude of prediciton errors
# Mean Absolute Erroe -> MAE calculates the average absolute difference between the predicted   and true values. It is less sensitive to outliers
# R Squared -  This measures the proportion of variance in the target variable that is explained by the model it ranges from 0 to 1 with higher values indicating better fit

# Mean Absolute Error in Python
#from sklearn.metrics import mean_absolute_error
#mae = mean_absolute_error(y_true, y_pred)   
#print("Mean Absolute Error: ", mae)


# Mean Squared Error in Python
#from sklearn.metrics import mean_squared_error
#mse = mean_squared_error(y_true, y_pred)
#print("Mean Squared Error: ", mse)


# Root Squared Error in Python
#from sklearn.metrics import mean_squared_error
#import numpy as np
#mse = mean_squared_error(y_true, y_pred)
#rmse = np.sqrt(mse)


# Multiple Linear Regression Example:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Read the dataset
dataset = pd.read_csv("E:\Data Science Env\ML\Startups.csv")

# Extract the features and target variable
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Perform one-hot encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Train the linear regression model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict the test set values
y_pred = regressor.predict(x_test)

# Create a scatter plot of the area and price data points
plt.scatter(x_test[:, 3], y_test, color='blue', label='Actual')
plt.scatter(x_test[:, 3], y_pred, color='red', label='Predicted')

# Print the intercept of the linear regression line
print("Intercept:", regressor.intercept_)

# Plotting the linear regression line
plt.plot(x_test[:, 3], regressor.predict(x_test), color='green', label='Regression Line')

# Plotting the new predictions
new_x = np.array([[0, 1, 0, 123456, 78901, 234567]])
new_y_pred = regressor.predict(new_x)
plt.scatter(new_x[:, 3], new_y_pred, color='orange', label='New Prediction')

# Display the legend
plt.legend()

# Show the plot
plt.show()

# Calculate the Root Mean Squared Error (RMSE)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", rmse)
