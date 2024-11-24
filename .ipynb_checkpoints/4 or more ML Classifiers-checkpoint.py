<<<<<<< HEAD
#!/usr/bin/env python
# coding: utf-8

# In[20]:


=======
# %%
>>>>>>> 0171a8a714cb374f69408ed415b976a597c5ba09
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
<<<<<<< HEAD
get_ipython().run_line_magic('matplotlib', 'inline')
=======
%matplotlib inline
>>>>>>> 0171a8a714cb374f69408ed415b976a597c5ba09
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

<<<<<<< HEAD

# Implement 1st ML Classifier: Random Forest

# In[21]:


=======
# %% [markdown]
# Implement 1st ML Classifier: Random Forest

# %%
>>>>>>> 0171a8a714cb374f69408ed415b976a597c5ba09
import pandas as pd

internet_data = pd.read_csv('preprocessed_internet_data.csv')

internet_data

<<<<<<< HEAD

# In[22]:


=======
# %%
>>>>>>> 0171a8a714cb374f69408ed415b976a597c5ba09
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

<<<<<<< HEAD

# In[23]:


X = internet_data.drop('Action', axis=1)  
y = internet_data['Action']


# In[24]:


=======
# %%
X = internet_data.drop('Action', axis=1)  
y = internet_data['Action']

# %%
>>>>>>> 0171a8a714cb374f69408ed415b976a597c5ba09
nat_ports = internet_data['NAT Source Port']
nat_ports

target_variable = y 

relationship_df = pd.DataFrame({
    'nat source port': nat_ports,
    'Target': target_variable
})

relationship_df

<<<<<<< HEAD

# In[25]:


print(y.value_counts())


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 1. Random Forest

# Let's first assess feature importance so we can figure out which features contribute the most to our model's performance

# In[27]:


=======
# %%
print(y.value_counts())

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# 1. Random Forest

# %% [markdown]
# Let's first assess feature importance so we can figure out which features contribute the most to our model's performance

# %%
>>>>>>> 0171a8a714cb374f69408ed415b976a597c5ba09
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=2)

# Stratified KFold cross-validation
skf = StratifiedKFold(n_splits=5)

# Store feature importances for each fold
feature_importances = []

cv_results = cross_validate(model, X, y, cv=skf, return_estimator=True)

for estimator in cv_results['estimator']:
    feature_importances.append(estimator.feature_importances_)

# Calculate and display mean metrics
mean_metrics = {key: np.mean(values) for key, values in cv_results.items() if key.startswith('test_')}
print("Average Metrics across folds:")
for metric, value in mean_metrics.items():
    print(f"  {metric.replace('test_', '').capitalize()}: {value:.4f}")

# Convert the list of feature importances to a DataFrame for easier interpretation
feature_importances = np.array(feature_importances)

# Average the feature importances across folds
mean_importances = feature_importances.mean(axis=0)

# Create a DataFrame with feature names and their corresponding importances
feature_names = X.columns  
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': mean_importances
})

# Sort by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print(importance_df)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importances in Random Forest (Average across folds)')
plt.show()

<<<<<<< HEAD

# Interpreting the feature importance graph, we see that the two most important features are NAT Source Port and Elapsed Time. Let's use random forest with a depth of 2 and a singular decision tree with a depth of 2 to test the accuracy of our model using these features. 

# In[28]:


=======
# %% [markdown]
# Interpreting the feature importance graph, we see that the two most important features are NAT Source Port and Elapsed Time. Let's use random forest with a depth of 2 and a singular decision tree with a depth of 2 to test the accuracy of our model using these features. 

# %%
>>>>>>> 0171a8a714cb374f69408ed415b976a597c5ba09
tree_model = DecisionTreeClassifier(max_depth=2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)

# Stratified KFold cross-validation
skf = StratifiedKFold(n_splits=5)

# Store metrics and confusion matrices for each fold
tree_accuracies, rf_accuracies = [], []
tree_conf_matrices, rf_conf_matrices = [], []

for train_index, test_index in skf.split(X, y):
    # Split data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train and predict with Decision Tree
    tree_model.fit(X_train, y_train)
    tree_preds = tree_model.predict(X_test)
    tree_accuracies.append(accuracy_score(y_test, tree_preds))
    tree_conf_matrices.append(confusion_matrix(y_test, tree_preds))

    # Train and predict with Random Forest
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_accuracies.append(accuracy_score(y_test, rf_preds))
    rf_conf_matrices.append(confusion_matrix(y_test, rf_preds))

# Average accuracies
avg_tree_accuracy = np.mean(tree_accuracies)
avg_rf_accuracy = np.mean(rf_accuracies)

print(f"Average Accuracy (Decision Tree, max_depth=2): {avg_tree_accuracy:.4f}")
print(f"Average Accuracy (Random Forest, n_estimators=100, max_depth=2): {avg_rf_accuracy:.4f}")

# Visualize confusion matrices for the last fold
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

ConfusionMatrixDisplay(tree_conf_matrices[-1], display_labels=tree_model.classes_).plot(ax=axes[0], cmap="Blues")
axes[0].set_title("Decision Tree Confusion Matrix (Last Fold)")

ConfusionMatrixDisplay(rf_conf_matrices[-1], display_labels=rf_model.classes_).plot(ax=axes[1], cmap="Blues")
axes[1].set_title("Random Forest Confusion Matrix (Last Fold)")

plt.tight_layout()
plt.show()

<<<<<<< HEAD

# We see that there is not much of a difference between the accuracy of the single decision tree and random forest. We can consider using one decision tree for this data since it computationally inexpensive, especially with a depth of 2. 

# 2. K-Nearest Neighbor

# In[29]:


=======
# %% [markdown]
# We see that there is not much of a difference between the accuracy of the single decision tree and random forest. We can consider using one decision tree for this data since it computationally inexpensive, especially with a depth of 2. 

# %% [markdown]
# 2. K-Nearest Neighbor

# %%
>>>>>>> 0171a8a714cb374f69408ed415b976a597c5ba09
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Range of k values to test
k_range = range(1, 21)  

# Store the accuracy for each k
knn_accuracies = []

# Stratified K-Fold Cross-Validation for each k value
for k in k_range:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    fold_accuracies = [] 

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        knn_model.fit(X_train, y_train)
        knn_preds = knn_model.predict(X_test)

        fold_accuracies.append(accuracy_score(y_test, knn_preds))

    # Average accuracy for this value of k
    knn_accuracies.append(np.mean(fold_accuracies))

# Plot the elbow curve to find the best value for k
plt.figure(figsize=(8, 6))
plt.plot(k_range, knn_accuracies, marker='o', color='b', linestyle='-', linewidth=2, markersize=8)
plt.title('KNN Elbow Plot (Accuracy vs. k)', fontsize=14)
plt.xlabel('Number of Neighbors (k)', fontsize=12)
plt.ylabel('Average Accuracy', fontsize=12)
plt.xticks(k_range)
plt.grid(True)
plt.show()

<<<<<<< HEAD

# The highest accuracy for k looks to be 7, although all the values for k seem to be above 98%. We will stick with k = 7 for this model.

# In[30]:


=======
# %% [markdown]
# The highest accuracy for k looks to be 7, although all the values for k seem to be above 98%. We will stick with k = 7 for this model.

# %%
>>>>>>> 0171a8a714cb374f69408ed415b976a597c5ba09
# Initialize KNN model with 7 neighbors
k_neighbors = 7
knn_model = KNeighborsClassifier(n_neighbors=k_neighbors)

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5)

knn_accuracies = []
knn_conf_matrices = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    knn_model.fit(X_train, y_train)
    knn_preds = knn_model.predict(X_test)

    knn_accuracies.append(accuracy_score(y_test, knn_preds))
    knn_conf_matrices.append(confusion_matrix(y_test, knn_preds))

# Average accuracy
avg_knn_accuracy = np.mean(knn_accuracies)

print(f"Average Accuracy (KNN, k={k_neighbors}): {avg_knn_accuracy:.4f}")

plt.figure(figsize=(6, 6))
ConfusionMatrixDisplay(knn_conf_matrices[-1], display_labels=knn_model.classes_).plot(cmap="Blues")
plt.title(f"KNN Confusion Matrix (Last Fold, k={k_neighbors})")
plt.show()

<<<<<<< HEAD

# 3.  Support Vector Machines

# In[31]:


=======
# %% [markdown]
# 3.  Support Vector Machines

# %%
>>>>>>> 0171a8a714cb374f69408ed415b976a597c5ba09
from sklearn.svm import SVC

svm_model = SVC()
svm_model.fit(X_train, y_train)
predictions = svm_model.predict(X_test)
#print(X_train, y_train)

acc_score = accuracy_score(y_test, predictions)
print(f'SVM accuracy score: {acc_score:.4f}')


<<<<<<< HEAD
# Because we get a high accuracy, we can verify this by doing k-fold cross validation

# In[32]:


=======
# %% [markdown]
# Because we get a high accuracy, we can verify this by doing k-fold cross validation

# %%
>>>>>>> 0171a8a714cb374f69408ed415b976a597c5ba09
# Stratified KFold cross-validation
skf = StratifiedKFold(n_splits=5)

svm_accuracies = []
svm_conf_matrices = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    svm_model.fit(X_train, y_train)
    svm_preds = svm_model.predict(X_test)

    svm_accuracies.append(accuracy_score(y_test, svm_preds))
    svm_conf_matrices.append(confusion_matrix(y_test, svm_preds))

# Average accuracy
avg_svm_accuracy = np.mean(svm_accuracies)

print(f"SVM Average Accuracy: {avg_svm_accuracy:.4f}")

plt.figure(figsize=(6, 6))
ConfusionMatrixDisplay(svm_conf_matrices[-1], display_labels=svm_model.classes_).plot(cmap="Blues")
plt.title(f"SVM Confusion Matrix")
plt.show()

<<<<<<< HEAD

# Using SVM and what we determined to be the two most important features, we can see that this version model gives up 79% accuracy

# In[33]:


=======
# %% [markdown]
# Using SVM and what we determined to be the two most important features, we can see that this version model gives up 79% accuracy

# %%
>>>>>>> 0171a8a714cb374f69408ed415b976a597c5ba09
from sklearn.svm import SVC
X2 = X[['NAT Source Port', 'Elapsed Time (sec)']] 
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=42)
svm_model = SVC()
svm_model.fit(X_train, y_train)
predictions = svm_model.predict(X_test)
#print(X_train, y_train)

acc_score = accuracy_score(y_test, predictions)
print(f'SVM accuracy score: {acc_score:.4f}')

<<<<<<< HEAD

# 4. Linear

# In[34]:


=======
# %% [markdown]
# 4. Linear

# %%
>>>>>>> 0171a8a714cb374f69408ed415b976a597c5ba09
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

score = linear_model.score(X_test, y_test)
print(f'linear model score: {score:.4f}')

<<<<<<< HEAD

# We can use K-fold cross validation to analyze our linear model. Here we see it has a high value for R² showing it explains a good amount of the variance in our data.

# In[35]:

=======
# %% [markdown]
# We can use K-fold cross validation to analyze our linear model. Here we see it has a high value for R² showing it explains a good amount of the variance in our data.

# %%
>>>>>>> 0171a8a714cb374f69408ed415b976a597c5ba09

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

# KFold cross-validation
kf = KFold(n_splits=5)

linear_accuracies = []
linear_conf_matrices = []
linear_rmse = []
linear_r2 = []

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    linear_model.fit(X_train, y_train)
    linear_preds = linear_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, linear_preds))
    r2 = r2_score(y_test, linear_preds)

    linear_rmse.append(rmse)
    linear_r2.append(r2)

avg_rmse = np.mean(linear_rmse)
avg_r2 = np.mean(linear_r2)

print(f"Average RMSE: {avg_rmse:.4f}")
print(f"Average R²: {avg_r2:.4f}")

# create a plot of the residuals vs predicted values
plt.figure(figsize=(6, 6))
plt.scatter(linear_preds, y_test - linear_preds, color='blue')
plt.axhline(y=0, color='r', linestyle='--')  # Reference line at y=0
plt.title("Residuals vs. Predicted Values")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals (Actual - Predicted)")
plt.show()

<<<<<<< HEAD

# In[ ]:


=======
# %%
>>>>>>> 0171a8a714cb374f69408ed415b976a597c5ba09
num_features = len(X.columns)
num_cols = 3
num_rows = (num_features + num_cols - 1) // num_cols  # Calculate number of rows to fit all features
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
axes = axes.flatten()


for idx, feature in enumerate(X.columns):
    
    linear_model.fit(internet_data[[feature]], y)
    y_pred = linear_model.predict(internet_data[[feature]])
  
    axes[idx].scatter(internet_data[feature], y, color='blue', label='Actual Data')
    axes[idx].plot(internet_data[feature], y_pred, color='red', label='Regression Line')
    
    axes[idx].set_title(f"Linear Regression: Action vs {feature}")
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel("Action")
    axes[idx].legend()

for idx in range(num_features, len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.show()

<<<<<<< HEAD

# 5. Logistic Regression

# In[ ]:


=======
# %% [markdown]
# 5. Logistic Regression

# %%
>>>>>>> 0171a8a714cb374f69408ed415b976a597c5ba09
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# do some standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# reduce the features so we can graph our model
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train the Multinomial Logistic Regression model
model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train_pca, y_train)

# use mesh grid to help plot diecision boundary
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# create the prediction areas
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# create the plot.
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, edgecolors='k', marker='o', cmap=plt.cm.Paired)
<<<<<<< HEAD
plt.title('Multinomial Logistic Regression Decision Boundary')
=======
plt.title('Logistic Regression')
>>>>>>> 0171a8a714cb374f69408ed415b976a597c5ba09
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar()
plt.show()

<<<<<<< HEAD
=======


>>>>>>> 0171a8a714cb374f69408ed415b976a597c5ba09
