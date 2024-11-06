#!/usr/bin/env python
# coding: utf-8

# Setup

# In[24]:


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

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "ml-predict-firewall-actions"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# Preview unedited data set

# In[25]:


import pandas as pd

internet_data = pd.read_csv('internet-firewall-data.csv')

internet_data


# Move 'Action' column to last column

# In[26]:


column_to_move = 'Action'

new_column_order = [col for col in internet_data.columns if col != column_to_move] + [column_to_move]

internet_data = internet_data[new_column_order]
internet_data2 = internet_data # copy for correlation matrix later


# Rename columns to have consistant format

# In[27]:


internet_data.rename(columns={'pkts_sent': 'Packets Sent', 'pkts_received': 'Packets Received'}, inplace=True)

internet_data


# We considered combining features but we already have the optimal amount of features for our data; combining features would result in lower accuracy.

# Discover and Visualize the data to gain insights

# In[28]:


internet_data.info()


# Based on our dataset info, there are no missing values that we need to consider.

# In[29]:


internet_data["Action"].value_counts()


# In[30]:


internet_data.describe()


# In[31]:


sns.set(style="whitegrid")

# Plot 1: Bytes Sent vs Bytes Received
plt.figure(figsize=(10, 6))
plt.title("Bytes Sent vs Bytes Received")
plt.xlabel("Bytes Sent")
plt.ylabel("Bytes Received")
plot = sns.scatterplot(x="Bytes Sent", y="Bytes Received", data=internet_data, hue="Action")
plot.set(ylim=(0, 80000000))
plot.set(xlim=(0, 500000))
plt.show()

# Plot 2: Elapsed Time (sec) vs Packets
plt.figure(figsize=(10, 6))
plt.title("Elapsed Time vs Number of Packets")
plt.xlabel("Elapsed Time (sec)")
plt.ylabel("Number of Packets")
plot = sns.scatterplot(x="Elapsed Time (sec)", y="Packets", data=internet_data, hue="Action")
plot.set(ylim=(0, 100000))
plt.show()

# Plot 3: Packet Sent vs Packet Received
plt.figure(figsize=(10, 6))
plt.title("Packets Sent vs Packets Received")
plt.xlabel("Packets Sent")
plt.ylabel("Packets Received")
plot = sns.scatterplot(x="Packets Sent", y="Packets Sent", data=internet_data, hue="Action")
plot.set(ylim=(0, 50000))
plot.set(xlim=(0, 20000))
plt.show()


# Use Label Encoding to convert the 'Action' feature from categorical to numerical

# In[32]:


internet_data['Action'] = internet_data['Action'].astype('category').cat.codes

# Calculate Pearson correlation
correlation_matrix1 = internet_data.corr(method='pearson')
correlation_matrix1


# Use One-Hot Encoding to convert the 'Action' feature from categorical to numerical

# In[33]:


internet_data_one_hot = pd.get_dummies(internet_data2, columns=['Action'])

# Calculate Pearson correlation
correlation_matrix2 = internet_data_one_hot.corr(method='pearson')
correlation_matrix2


# Correlation Matrix Heatmap

# In[34]:


import seaborn as sns
import matplotlib.pyplot as plt

# Show heatmap for one-hot encoding correlation matrix
plt.figure(figsize=(15, 15))
plt.title("Correlation Matrix - one-hot encoding version")
sns.heatmap(correlation_matrix2, annot=True, cmap='coolwarm')
plt.show()

# Show heatmap for label encoding correlation matrix
plt.figure(figsize=(15, 15)
plt.title("Correlation Matrix - label encoding version")
sns.heatmap(correlation_matrix1, annot=True, cmap='coolwarm')
plt.show()


# Verify that 'Bytes' equals 'Bytes Sent' plus 'Bytes Received' so we can eliminate this column.
# This is necessary so we don't have extra weight in our data based on the Bytes features.

# In[35]:


internet_data_new = internet_data[['Bytes Sent', 'Bytes Received', 'Bytes']].copy()
internet_data_new['Cal Total Bytes'] = internet_data_new['Bytes Sent'] + internet_data_new['Bytes Received']
internet_data_new['Valid Bytes'] = internet_data_new['Cal Total Bytes'] == internet_data_new['Bytes']
internet_data_new


# Verify that 'Packets' equals 'Packets Sent' plus 'Packets Sent' so we can eliminate this column.
# This is necessary so we don't have extra weight in our data based on the Packets features.

# In[36]:


internet_data_new = internet_data[['Packets Sent', 'Packets Received', 'Packets']].copy()
internet_data_new['Cal Total Packets'] = internet_data_new['Packets Sent'] + internet_data_new['Packets Received']
internet_data_new['Valid Packets'] = internet_data_new['Cal Total Packets'] == internet_data_new['Packets']
internet_data_new


# In[37]:


internet_data = internet_data.drop('Bytes', axis=1)
internet_data = internet_data.drop('Packets', axis=1)
internet_data


# In[47]:


correlation_matrix1 = internet_data.corr(method='pearson')
import seaborn as sns
import matplotlib.pyplot as plt

# Show heatmap for one-hot encoding correlation matrix
plt.figure(figsize=(15, 15))
sns.heatmap(correlation_matrix1, annot=True, cmap='coolwarm')
plt.show()


# Plot features that may need to be scaled/normalized

# In[40]:


columns_to_plot = ['Bytes Sent', 'Bytes Received']

plt.figure(figsize=(15, 6))
sns.boxplot(data=internet_data[columns_to_plot])
plt.title('Box Plot for Bytes Sent and Bytes Received')
plt.show()


# Implement Standardization and Normalization for Bytes Sent and Bytes Received

# In[41]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler

columns_to_scale = ['Bytes Sent', 'Bytes Received']

# 1. Standardization
scaler = StandardScaler()
standardized_data = scaler.fit_transform(internet_data[columns_to_scale])
standardized_df = pd.DataFrame(standardized_data, columns=columns_to_scale)

# 2. Normalization
normalizer = MinMaxScaler()
normalized_data = normalizer.fit_transform(internet_data[columns_to_scale])
normalized_df = pd.DataFrame(normalized_data, columns=columns_to_scale)

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.boxplot(data=standardized_df)
plt.title('Standardized Box Plot for Bytes Sent and Bytes Received')

plt.subplot(1, 2, 2)
sns.boxplot(data=normalized_df)
plt.title('Normalized Box Plot for Bytes Sent and Bytes Received')

plt.tight_layout()
plt.show()


# In[42]:


columns_to_plot = ['Packets Sent', 'Packets Received']

plt.figure(figsize=(15, 6))
sns.boxplot(data=internet_data[columns_to_plot])
plt.title('Box Plot for Packets Sent and Packets Received')
plt.show()


# Implement Standardization for Packets Sent and Packets Received

# In[43]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler

columns_to_scale = ['Packets Sent', 'Packets Received']

# 1. Standardization
scaler = StandardScaler()
standardized_data = scaler.fit_transform(internet_data[columns_to_scale])
standardized_df = pd.DataFrame(standardized_data, columns=columns_to_scale)

# 2. Normalization
normalizer = MinMaxScaler()
normalized_data = normalizer.fit_transform(internet_data[columns_to_scale])
normalized_df = pd.DataFrame(normalized_data, columns=columns_to_scale)

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.boxplot(data=standardized_df)
plt.title('Standardized Box Plot for Packets Sent and Packets Received')

plt.subplot(1, 2, 2)
sns.boxplot(data=normalized_df)
plt.title('Normalized Box Plot for Packets Sent and Packets Received')

plt.tight_layout()
plt.show()


# Perform log transformation on the bytes sent and bytes received

# In[44]:


internet_data_logged_data = internet_data.copy()
internet_data_logged_data['Log Bytes Sent'] = np.log1p(internet_data['Bytes Sent'])
internet_data_logged_data['Log Bytes Received'] = np.log1p(internet_data['Bytes Received'])

log_columns_to_plot = ['Log Bytes Sent', 'Log Bytes Received']

plt.figure(figsize=(15,6))
sns.boxplot(data=internet_data_logged_data[log_columns_to_plot])
plt.title('Box Plot for Log-Transformed Bytes')
plt.show()


# Perform log transformation on packets sent and packets received

# In[45]:


internet_data_logged_data['Log Packets Sent'] = np.log1p(internet_data['Packets Sent'])
internet_data_logged_data['Log Packets Received'] = np.log1p(internet_data['Packets Received'])

log_columns_to_plot = ['Log Packets Sent', 'Log Packets Received']

plt.figure(figsize=(15,6))
sns.boxplot(data=internet_data_logged_data[log_columns_to_plot])
plt.title('Box Plot for Log-Transformed Packets')
plt.show()


# We notice that there are still outliers after performing a log transofrmation of the bytes sent and bytes received, as well as packets sent and packets received. Since we're analyzing internet traffic data, very large or very small values (in terms of bytes sent/received) might represent legitimate high-usage events or anomalies. Removing or altering these could distort the real insights.

# Plot a normal distribution curve on the logged bytes data for further visualization.

# In[46]:


from scipy.stats import norm

# Fit a normal distribution for the log-transformed 'Bytes Sent'
mean_log_sent, std_log_sent = norm.fit(internet_data_logged_data['Log Bytes Sent'])

# Fit a normal distribution for the log-transformed 'Bytes Received'
mean_log_received, std_log_received = norm.fit(internet_data_logged_data['Log Bytes Received'])

# Create a range of values from min to max for the log-transformed data
x_log_sent = np.linspace(internet_data_logged_data['Log Bytes Sent'].min(), internet_data_logged_data['Log Bytes Sent'].max(), 100)
x_log_received = np.linspace(internet_data_logged_data['Log Bytes Received'].min(), internet_data_logged_data['Log Bytes Received'].max(), 100)

# Calculate the PDF for log-transformed data
pdf_log_sent = norm.pdf(x_log_sent, mean_log_sent, std_log_sent)
pdf_log_received = norm.pdf(x_log_received, mean_log_received, std_log_received)

plt.figure(figsize=(15, 6))

plt.plot(x_log_sent, pdf_log_sent, label='Fitted Normal - Log Bytes Sent', color='blue')

plt.plot(x_log_received, pdf_log_received, label='Fitted Normal - Log Bytes Received', color='orange')

plt.title('Normal Distribution Curve for Log-Transformed Bytes Sent and Bytes Received')
plt.xlabel('Log Bytes')
plt.ylabel('Probability Density')
plt.legend()
plt.show()

