#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 14:25:57 2018

@author: Naresh Kumar
"""
# =============================================================================
# *************Problem Statement****************
# -In this project you are going to predict the overall rating of soccer player based on their attributes
# such as 'crossing', 'finishing etc.
# 
# -Explain the accuracy of Model
# =============================================================================


# =============================================================================
# ****************PSEUDOCODE*********************
# -Import the data 
# -Preprocess the data
# -Split the Data to Train and Test Data 
# -Convert Categorical values to binary values 
# -Build a Multi Linear Regression Model
# -Fit the Model
# -Find the accuracy and Variable weight using OLS(Ordinary Least Square)
# =============================================================================

import sqlite3 as db
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

#Preparing Dataset

cnx = db.connect('database.sqlite')
df = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)

df.columns.tolist()

df=df.reindex_axis([ 'overall_rating',
 'id',
 'player_fifa_api_id',
 'player_api_id',
 'date',
 'potential',
 'preferred_foot',
 'attacking_work_rate',
 'defensive_work_rate',
 'crossing',
 'finishing',
 'heading_accuracy',
 'short_passing',
 'volleys',
 'dribbling',
 'curve',
 'free_kick_accuracy',
 'long_passing',
 'ball_control',
 'acceleration',
 'sprint_speed',
 'agility',
 'reactions',
 'balance',
 'shot_power',
 'jumping',
 'stamina',
 'strength',
 'long_shots',
 'aggression',
 'interceptions',
 'positioning',
 'vision',
 'penalties',
 'marking',
 'standing_tackle',
 'sliding_tackle',
 'gk_diving',
 'gk_handling',
 'gk_kicking',
 'gk_positioning',
 'gk_reflexes'],axis=1)


# Preprocessing the Dataset 
df.preferred_foot.unique()
df.attacking_work_rate.unique()
df.defensive_work_rate.unique()


df.isnull().any().any()
df=df.dropna()

df=df.loc[df['preferred_foot'].isin(['right', 'left']) & df.attacking_work_rate.isin(['medium', 'high', 'low']) 
& df.defensive_work_rate.isin(['medium', 'high', 'low'])]


df=df.drop(df.columns[[1, 2, 3, 4]], axis=1)

df.describe()

# Splitting Dependent and Independent variables 
X=pd.DataFrame(df.iloc[: , 1:].values)
y=df.iloc[:, 0].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_1 = LabelEncoder()
X.iloc[:, 1] = labelencoder_x_1.fit_transform(X.iloc[:, 1])
labelencoder_x_2 = LabelEncoder()
X.iloc[:, 2] = labelencoder_x_2.fit_transform(X.iloc[:, 2])
labelencoder_x_3= LabelEncoder()
X.iloc[:, 3] = labelencoder_x_3.fit_transform(X.iloc[:, 3])
onehotencoder_1 = OneHotEncoder(categorical_features = [1])
X = onehotencoder_1.fit_transform(X).toarray()
onehotencoder_2 = OneHotEncoder(categorical_features = [3])
X = onehotencoder_2.fit_transform(X).toarray()
onehotencoder_3 = OneHotEncoder(categorical_features = [6])
X = onehotencoder_3.fit_transform(X).toarray()



# Avoiding the Dummy Variable Trap
X=np.delete(X, [0,2,5], axis=1)



# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)



# Predicting the Test set results
y_pred = regressor.predict(X_test)



#Building the optimal model using Backward Elimination
import statsmodels.api as sm
X= sm.add_constant(X)
X_opt=X[:, :]
model = sm.OLS(y, X_opt).fit() ## sm.OLS(output, input)
predictions = model.predict(X_opt)
model.summary()

X_opt=np.delete(X,[30,33], axis=1)
model = sm.OLS(y, X_opt).fit() ## sm.OLS(output, input)
predictions = model.predict(X_opt)
model.summary()

# =============================================================================
# 
# conclusion: -Model explains ~85 % of variance. 
#             -OLS method is used determine how much variance is expalained
#             -R Square and Adjusted R Square value is used as parameter to optimise the model accuracy
#             -All the Variables are used except for ['id','player_fifa_api_id','player_api_id','date'] for 
#              building Model
#             -OLS method is used eliminate Varibles but from model it is known all the variables are needed for accuracy
#             -Hence model acquired ~85% accuracy
# =============================================================================
            
            
            

























