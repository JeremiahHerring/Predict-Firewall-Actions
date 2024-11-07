#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np
import os

import seaborn as sns
from scipy import stats

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


# In[11]:


import pandas as pd

internet_data = pd.read_csv('preprocessed_internet_data.csv')

internet_data


# Implement Linear Regression Preproccessed Data

# Prepare features and target variable

# In[12]:


X = internet_data.drop(columns=['Action'])
y = internet_data['Action']


# Split the data into training and testing sets

# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[14]:


from sklearn.linear_model import LinearRegression
import pickle

model = LinearRegression()
model.fit(X_train, y_train)

filename='finalized_model_M1.sav'
pickle.dump(model, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))

score = loaded_model.score(X_test, y_test)
print(score)


# Implement Linear Regression on Original Full Feature Set

# In[15]:


original_internet_data = pd.read_csv('internet_data_label_encoding.csv')

original_internet_data


# In[16]:


original_x = original_internet_data.drop(columns=['Action'])
original_y = original_internet_data['Action']


# In[17]:


original_x_train, original_x_test, original_y_train, original_y_test = train_test_split(original_x, original_y, test_size=0.3, random_state=42)


# In[18]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(original_x_train, original_y_train)

filename='finalized_model_original_M1.sav'
pickle.dump(model, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))

score = loaded_model.score(original_x_test, original_y_test)
print(score)


# We see that our linear regression model accuracy with the preprocessed data is greater than the model accuracy with the original, full feature set, implying that data pre-processing helped improve model accuracy. 
