# Recommender Sytem
# A recommender system algorithm is a method or approach used to create recommendations or predictions for users based on their preferences or behavior

# There are various types of recommender system algorithms, including:

# 1. Collaborative Filtering: Collaborative filtering algorithms recommend items based on the preferences and behavior of similar users. They find patterns and similarities between users' interactions with items to make predictions. This can be done through user-based or item-based collaborative filtering.

# 2. Content-Based Filtering: Content-based filtering algorithms recommend items based on their attributes or content features. They analyze the characteristics or descriptions of items and suggest similar items to what the user has previously shown interest in.

# 3. Hybrid Methods: Hybrid methods combine multiple approaches, such as collaborative filtering and content-based filtering, to leverage the strengths of different techniques. These methods provide more accurate and diverse recommendations by combining different sources of information.

# 4. Matrix Factorization: Matrix factorization algorithms factorize a user-item interaction matrix to discover latent factors or features that represent user preferences and item characteristics. These latent factors are then used to generate recommendations.

# 5. Deep Learning Approaches: Deep learning algorithms, such as neural networks, can be used for recommendation tasks. These models learn complex patterns and representations from large amounts of data to provide personalized recommendations

# Example:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
%matplotlib inline

# Define column names and read the user ratings data
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv(r"E:\Data Science Env\ML\u_data.data", sep='\t', names=column_names)
df.head()

# Read the movie titles data
movie_title = pd.read_csv(r"E:\Data Science Env\ML\Movie_Id_Titles")
movie_title.head()

# Merge the user ratings data with the movie titles data
df = pd.merge(df, movie_title, on='item_id')
df.head()

# Count the number of ratings for each movie
df.groupby('title')['rating'].count().sort_values(ascending=False).head()

# Calculate the mean rating for each movie
df.groupby('title')['rating'].mean().sort_values(ascending=False).head()
df.head()

# Create a DataFrame for ratings
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings.head()

# Add a column for the number of ratings
ratings['num of ratings'] = df.groupby('title')['rating'].count()
ratings.head()

# Visualize the distribution of the number of ratings
plt.figure(figsize=(10, 4))
ratings['num of ratings'].hist(bins=70)

# Visualize the distribution of the ratings
plt.figure(figsize=(10, 4))
ratings['rating'].hist(bins=70)

# Visualize the relationship between rating and number of ratings
sns.jointplot(x='rating', y='num of ratings', data=ratings, alpha=0.5)

# Create a pivot table for collaborative filtering
moviemat = df.pivot_table(index='user_id', columns='title', values='rating')
moviemat.head()

# Get the top movies based on the number of ratings
ratings.sort_values('num of ratings', ascending=False).head(10)

# Extract user ratings for a specific movie ("Star Wars (1977)")
starwars_user_ratings = moviemat['Star Wars (1977)']
starwars_user_ratings.head()

# Extract user ratings for a specific movie ("Liar Liar (1997)")
liarliar_user_ratings = moviemat["Liar Liar (1997)"]
liarliar_user_ratings.head()

# Calculate the correlation between "Star Wars (1977)" and other movies
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)

# Calculate the correlation between "Liar Liar (1997)" and other movies
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)

# Create a DataFrame for the correlation with "Star Wars (1977)"
corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars.head()

# Sort the movies by correlation with "Star Wars (1977)"
corr_starwars.sort_values('Correlation', ascending=False).head(10)

# Join the correlation values with the number of ratings
corr_starwars = corr_starwars.join(ratings['num of ratings'])
corr_starwars.head()

# Filter movies with more than 100 ratings and sort by correlation
corr_starwars[corr_starwars['num of ratings'] > 100].sort_values('Correlation', ascending=False).head()
