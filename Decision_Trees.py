# Decision Trees

# Decision trees are a popular intuitive machine learning algorithm used for both classification and regression tasks
# Decision trees are a type of supervised learning algorithm that can be used for both categorical and numerical input features


# Decision Trees Features

# Root Node -> The root node is the top most node in the tree and represents the starting point or descision making 
# The Root Node corresponds to the feature or attribute that provides the most signinificant split or descrimination power

# Internal Nodes -> These are decision points within the tree 
# Each internal node corresponds to a specific feature or attribute and a decision rule based on its values
# The decision rule determines which branch to follow based on the feature value of the instance being evaluated

# Branches -> Branches emanate from the internal code and represent the possible outcomes or values of the feature 
# Instances are directed along different branches based on the feature values, moving towards the appropriate node

# Leaf Nodes -> These are terminal nodes of the decision tree 
# They represent final outcomes or desicions made by the trees
# Each leaf node corresponds to a specific class label or a class distributiom

# Paths -> Paths in a decision tree represent the sequence of decisions to a particular eaf nodes
# Each path corresponds to a set of conditions that lead to a specific prediction or classification


# The Learning Algorithm of Decision Trees
# The Learning Algorithm of decision trees involves constructing the tree by recursively partitioning  the data  based on features

# Selecting the Root Node -> The Algorithm starts by selecting the best feature to act as the root node of the decision tree
# This feature is chosen based on certain criteria such as Gini impurity, entropy or information gain   
# The goal is to find the most useful information for classifying the data

# Splitting the Data -> Once the root node is selected, the dataset is partitioned into subsets based on the values of the chosen feature
# Each subset corresponds to a branch stemming from the root node 
# The algorithm creates child nodes for each unique value of the selected feature and assigns the corresponding subset  of data to each child node

# Handling Internal Nodes -> The process continues recursively for each child node treating them as internal nodes 
# At each internal node, each algorithm selects the best feature to split the data further, following the same criteria used for the root node
# This recursive splitting process continues until  stopping criterion is met such as reaching a maximum depth or a minimum number of   samples per leaf

# Creating Leaf Nodes -> When the stopping criterion is met, he algorithm creates leaf nodes 
# Each leaf node represents a class label or a numerical value and serves as the final prediciton for the corresponding subset of data  
# The class label assigned to a leaf node is typically determined by the majority   class of the smaples with that node

# Prunning -> This is optional
# After the decision is constructed, prunning can be applied to avoid overfitting 
# Prunning involves removing or merging nodes to simplify the tree while  maintaining its predictive accuracy 
# Various prunning techniques such as cost complexity prunning(also known as minimal cost complexity prunning or alpha-beta prunning) can be employed to optimize the trees structure 

# The learning algorithm of decision trees aims to create a tree that maximizes predictive accuracy while minimizing complexity

# Entropy
# Entropy is a concept from information theory that measures the impurity or uncertainity of a random variable
# In the context f decision trees, entropy is used as a criterion to determine the quality  of a split when selecting features for partitioning the data
# In decision trees, the goal is to find features that result in splits that maximize the information gain or decrease in entropy

# Gini Impurity
# Gini Impurity is a measure of the impurity or disorder of a node in a decision tree
# It quantifies the probability of misclassifying a randomly chosen element in a node if it were  randomly labeled according to the distribution of classes in that node
# When constructing  decision tree using the Gini impurity criterion, the algorithm aims to minimize the Gini impurity at each node
# It selects the feauture and threshold that results in the greatest reduction in Gini impuirity
# The process is repeated recursively to build a tree with nodes that progressively become more pure until a stopping criterion is met

# Information Gain
# In the context of decision trees, information gain is a metric used to determine the usefulness of a feature in splitting the data to create a more informative tree
# It measure reduction in entropy or impurity achived by splitting the data based on a particular feature



# from sklearn.tree import DecisionTreeClassifier
# i.e this is a Python code that imports the DecisionTreeClassifier class from the scikit learn library
# The DecisionTreeClassifier is a class that represents the implementation of the decision tree algorithm for classification taska
# It can used to create a decision tree model that can be trained on labelled data and used for making predicitons on new instances


# from sklearn.model_selection import train_test_split
# i.e the Python code imports the train_test_split function from the model_selection module of the sklearn library
# in ML, it is a common practice to split a given dataset into two subsets - one for training the model and the other for testing its performance on unseen data based on specified ratio or size


# from sklearn import metrics
# i.e the Python code imports the metrics module from the scikit-learn library
# the metrics metrics module provide various functions for evaluating the performance of machine learning models
# that allow you to calculate different metrics to assess the accuracy, precision, recall, F1-score, and other performance measures of your model


# import pandas as pd

# data = pd.read_csv("E:\Data Science Env\ML\salaries.csv", header= None, names= col_names)
# i.e header=None indicates that the CSV file does not have a header and thus the column names needs to specified seperately
# names=col_names assigns the provided list of column names to DataFrame  columns

# from sklearn import preprocessing
# i.e the python code imports the preprocessing module from the sklearn library
# the preprocessing module is a sub-module in sci-kit learn than provides various functions and  classes for data preprocessing
# It includes methods for transforming  and scalling data, encoding categorical variables, handling missing values and more


# label_encoder = preprocessing.LabelEncoder()
# i.e the Python code   creates an instance of the LabelEncoder class from the preprocessing module in sci-kit learn 
# LabelEncoder is a class used for encoding categorical labels into numerical values 
# It assigns a unique integer to each unique category or label  in the target variable

# data['company'] = label_encoder.fit_transform(data['company'])

# fit_transform -> this method fits the label_encoder to the unique values in the 'company' column of the data in the DataFrame
# This step learns the mapping between the unique categories in the column and assigns  a unique integer to each category
# Then it transforms the values int the respective column of the data in the DataFrame  by replacing each categorical value with its corresponding integer label 

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state = 100)
# i.e x_train represents the training set of input variables, the x_test represents the testing set of input variables, y_train represents the training set of the target variable, the y_test represents the testing set of the target variable
# train_test_split(x, y, test_size = 0.2, random_state = 100) splits the input variables 'x' and the target variables 'y' into training and testing test
# the test_size parameter specifies the proportion of the dataset that should be allocated to testing, as in this case is 20%
# The random_state parameter is used to set a seed value for the random number generator which  ensures reproducibility of the split
# After this line is executed, you will have four datasets, x_train, x_test, y_train, y_test


# clf_entropy = DecisionTreeClassifier(criterion = "entropy", max_depth = 3
# i.e DecisionTreeClassifier -> This is the class in scikit-learn that represents a decision tree classifier
# criterion = 'entropy -> specifies the measure used for splitting the nodes in the decision trees
# max_depth = 3 -> This parameter sets the maximum depth levels of the decision tree 
# It limits the splits and helps control the complexity of the tree
# In this case, the maximum depth is set to 3, which means  the tree will have a maximum of 3 levels from the root node


# clf_entropy = clf_entropy.fit(x_train, y_train)
# This is fitting the decision tree classifier "clf_entropy" to the training data x_train and the corresponding target variable y_train 
# clf_entropy -> This is the decision tree classifier object that was previously created using the DecisionTreeClassifier class
# fit(x_train, y_train) -> The fit method is used to train the decision tree classifier on the provided  training data

# Example:

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Define the column names for the dataset
col_names = ['company', 'job', 'degree', 'salary_more_than_100k']

# Read the dataset from a CSV file
data = pd.read_csv("E:\Data Science Env\ML\salaries.csv")

# Read the dataset from a CSV file with custom column names
data = pd.read_csv("E:\Data Science Env\ML\salaries.csv", header=None, names=col_names)

# Define the feature columns and target variable
feature_cols = ['company', 'job', 'degree']
x = data[feature_cols]
y = data['salary_more_than_100k']

# Import the preprocessing module from sklearn
from sklearn import preprocessing

# Initialize the label encoder
label_encoder = preprocessing.LabelEncoder()

# Encode the 'company', 'job', and 'degree' columns with label encoder
data['company'] = label_encoder.fit_transform(data['company'])
data['job'] = label_encoder.fit_transform(data['job'])
data['degree'] = label_encoder.fit_transform(data['degree'])

# Print the first few rows of the dataset
print(data.head())

# Reassign the feature columns and target variable after encoding
feature_cols = ['company', 'job', 'degree']
x = data[feature_cols]
y = data['salary_more_than_100k']

# Print the feature data and target variable
print(x)
print(y)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Create a DecisionTreeClassifier with entropy as the criterion and maximum depth of 3
clf_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Fit the decision tree classifier to the training data
clf_entropy = clf_entropy.fit(x_train, y_train)

# Use the trained classifier to make predictions on the test data
y_pred = clf_entropy.predict(x_test)

# Print the accuracy of the model by comparing the predicted and actual values
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))


# Example 2:

# (TO BE UPDATED WITH TIME)


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import graphviz
from sklearn.tree import export_graphviz

# Read the dataset
col_names = ['company', 'job', 'degree', 'salary_more_than_100k']
data = pd.read_csv("E:\Data Science Env\ML\salaries.csv")
data = pd.read_csv("E:\Data Science Env\ML\salaries.csv", header=None, names=col_names)

# Preprocess the data
feature_cols = ['company', 'job', 'degree']
x = data[feature_cols]
y = data['salary_more_than_100k']

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
data['company'] = label_encoder.fit_transform(data['company'])
data['job'] = label_encoder.fit_transform(data['job'])
data['degree'] = label_encoder.fit_transform(data['degree'])

print(data.head())

# Split the data into training and test sets
feature_cols = ['company', 'job', 'degree']
x = data[feature_cols]
y = data['salary_more_than_100k']
print(x)
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Train the decision tree classifier
clf_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf_entropy = clf_entropy.fit(x_train, y_train)

# Make predictions on the test set
y_pred = clf_entropy.predict(x_test)
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))

# Visualize the decision tree
dot_data = export_graphviz(clf_entropy, out_file=None, feature_names=feature_cols,
                           class_names=['No', 'Yes'], filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree")

# Evaluate the model performance
print('Precision:', metrics.precision_score(y_test, y_pred))
print('Recall:', metrics.recall_score(y_test, y_pred))
print('F1 Score:', metrics.f1_score(y_test, y_pred))

# Fine-tune hyperparameters using grid search
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 3, 4]
}

grid_search = GridSearchCV(estimator=clf_entropy, param_grid=param_grid, cv=5)
grid_search.fit(x_train, y_train)

print('Best Parameters:', grid_search.best_params_)
print('Best Score:', grid_search.best_score_)
print('Best Model:', grid_search.best_estimator_)
