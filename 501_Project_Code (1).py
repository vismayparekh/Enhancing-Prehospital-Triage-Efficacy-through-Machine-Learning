#!/usr/bin/env python
# coding: utf-8

# In[32]:


# Importing the required libraries
import pandas as pd
import collections
import numpy as np
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# In[33]:


# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming your data is stored in a DataFrame named 'df'
df = pd.read_csv("501_project_dataset.tsv", sep='\t')

# Original and custom column names mapping
column_mapping = {
    'Age': 'Age',
    'SBP': 'Systolid blood pressure',
    'DBP': 'Diastolic blood pressure',
    'HR': 'Heart rate',
    'RR': 'Respiration rate',
    'BT': 'Body temperature',
    'Saturation': 'Saturation to use pulse oximeter',
    'NRS_pain': 'Numeric rating scales of pain'
}

# Selecting a subset of numerical columns for analysis using custom names
numerical_columns_subset = list(column_mapping.values())

# Scatter plots
pair_plot = sns.pairplot(df[list(column_mapping.keys())])

# Customizing y-axis labels with custom names
for i, col in enumerate(numerical_columns_subset):
    pair_plot.axes[i, 0].set_ylabel(col)

# Adding a title to the pair plot
plt.suptitle('Pair Plot of Selected Numerical Variables', y=1.02)

# Display the plot
plt.show()


# In[34]:


# Set the age groups
bins = [20, 30, 40, 50, 60, 70, 80, 90, 100]
labels = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-100']

# Create a new column 'AgeGroup' based on age bins
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# Define a custom color palette
colors = sns.color_palette('husl', n_colors=len(labels))

# Histogram for Age-wise SBP comparison
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='SBP', hue='AgeGroup', multiple='stack', palette=colors)
plt.title('Age-wise Systolid blood pressure Comparison')
plt.xlabel('Systolid blood pressure')
plt.ylabel('Systolid blood pressure-Frequency')
plt.show()

# Histogram for Age-wise DBP comparison
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='DBP', hue='AgeGroup', multiple='stack', palette=colors)
plt.title('Age-wise Diastolic blood pressure Comparison')
plt.xlabel('Diastolic blood pressure')
plt.ylabel('Diastolic blood pressure-Frequency')
plt.show()


# In[35]:


# Demographic Analysis: Compare the distribution of patients across different types of Emergency Departments
plt.figure(figsize=(10, 6))
sns.countplot(x='Group', data=df, palette="Set1")
plt.title('Distribution of Patients Across Emergency Departments')
plt.xlabel('Emergency Department Type')
plt.ylabel('Count')
plt.xticks([0, 1], ['Local ED', 'Regional ED'])
plt.show()

# Demographic Analysis: Analyze the distribution of patients by gender
plt.figure(figsize=(8, 6))
sns.countplot(x='Sex', data=df, palette="Set2")
plt.title('Distribution of Patients by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks([0, 1], ['Female', 'Male'])
plt.show()

# Demographic Analysis: Explore the age distribution of patients
plt.figure(figsize=(12, 6))
sns.histplot(df['Age'], bins=20, kde=True, color='skyblue')
plt.title('Age Distribution of Patients')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[36]:


df = pd.read_csv("501_project_dataset.tsv", sep='\t')

# Mapping of arrival mode codes to labels
arrival_mode_labels = {
    1: 'Walking',
    2: 'Private car',
    3: 'Private ambulance',
    4: 'Public transportation',
    5: 'Wheelchair',
    6: 'Others'
}

# Replace arrival mode codes with labels
df['Arrival mode'] = df['Arrival mode'].map(arrival_mode_labels)

# Set the style for seaborn
sns.set(style="whitegrid")

# Arrival Mode Analysis: Compare the frequency of different arrival modes
plt.figure(figsize=(12, 6))
sns.countplot(x='Arrival mode', data=df, palette='viridis')
plt.title('Frequency of Different Arrival Modes')
plt.xlabel('Arrival Mode')
plt.ylabel('Count')
plt.show()

# Arrival Mode Analysis: Analyze the distribution of patients by walking, private car, ambulance, etc.
selected_arrival_modes = ['Walking', 'Private car', 'Private ambulance', 'Public transportation', 'Wheelchair', 'Others']
filtered_df = df[df['Arrival mode'].isin(selected_arrival_modes)]

plt.figure(figsize=(12, 6))
sns.countplot(x='Arrival mode', data=filtered_df, palette='viridis')
plt.title('Distribution of Patients by Selected Arrival Modes')
plt.xlabel('Arrival Mode')
plt.ylabel('Count')
plt.show()


# In[37]:


# Injury Analysis: Compare the distribution of patients with and without injury
plt.figure(figsize=(8, 5))
ax = sns.countplot(x='Injury', data=df, palette='viridis')

#  x-axis labels
injury_labels = {
    1: 'Non-injury',
    2: 'Injury'
}
ax.set_xticklabels([f"{value}" for key, value in injury_labels.items()])

plt.title('Distribution of Patients with and without Injury')
plt.xlabel('Injury')
plt.ylabel('Count')
plt.show()

# Injury Analysis: Explore the top 10 types of injuries reported
top_injuries = df['Cheif_Complain_Translated'].value_counts().nlargest(10).index

plt.figure(figsize=(12, 6))
sns.countplot(x='Cheif_Complain_Translated', hue='Injury', data=df[df['Cheif_Complain_Translated'].isin(top_injuries)], palette='viridis')
plt.title('Top 10 Types of Injuries Reported')
plt.xlabel('Chief Complaint (Injury)')
plt.ylabel('Count')
plt.legend(title='Injury', loc='upper right', labels=['No Injury', 'With Injury'])
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.show()


# In[38]:


# Binning the 'Age' column into different age groups
bins = [0, 18, 30, 50, 100]
labels = ['0-18', '19-30', '31-50', '51-100']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# Mental State Analysis: Analyze the distribution of mental states by age group
plt.figure(figsize=(14, 8))
ax = sns.countplot(x='Mental', hue='Age_Group', data=df, palette='Set2')

# x-axis labels for mental states
mental_state_labels = {
    1: 'Alert',
    2: 'Verbal response',
    3: 'Pain response',
    4: 'Unconsciousness'
}
ax.set_xticklabels([f"{value}" for key, value in mental_state_labels.items()])

# Setting the title and axis labels for the plot
plt.title('Distribution of Mental States by Age Group')
plt.xlabel('Mental State')
plt.ylabel('Count')
plt.show()

# Distribution of Pain among Patients
plt.figure(figsize=(12, 6))
ax = sns.countplot(x='Pain', data=df, palette='Set1')

# x-axis labels for pain categories
pain_labels = {
    0: 'Pain',
    1: 'Non-pain'
}
ax.set_xticklabels([f"{value}" for key, value in pain_labels.items()])

# Setting the title and axis labels for the plot
plt.title('Distribution of Patients with Pain and Without Pain')
plt.xlabel('Pain')
plt.ylabel('Count')
plt.show()


# In[39]:


vital_signs = ['SBP', 'DBP', 'HR', 'RR', 'BT', 'Saturation']
vital_signs_labels = ['Systolid blood pressure', 'Diastolic blood pressure', 'Heart rate', 'Respiration rate', 'Body temperature', 'Saturation to use pulse oximeter']

plt.figure(figsize=(18, 10))
for i, (vital_sign, label) in enumerate(zip(vital_signs, vital_signs_labels), 1):
    plt.subplot(2, 3, i)
    sns.histplot(data=df, x=vital_sign, kde=True, color='skyblue', bins=30)
    plt.title(f'Distribution of {label}')
    plt.xlabel(label)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Vital Signs Analysis: Explore the relationship between pain (NRS_pain) and vital signs
plt.figure(figsize=(18, 10))
for i, (vital_sign, label) in enumerate(zip(vital_signs, vital_signs_labels), 1):
    plt.subplot(2, 3, i)
    sns.lineplot(x=vital_sign, y='NRS_pain', data=df, color='coral', alpha=0.5)
    plt.title(f'Relationship between {label} and Numeric rating scales of pain')
    plt.xlabel(label)
    plt.ylabel('Numeric rating scales of pain')

plt.tight_layout(h_pad=2.5)  # Increase the height space between subplots
plt.show()


# In[40]:


# Mapping of KTAS codes to labels
ktas_labels = {
    1: 'Extremely Critical',
    2: 'Emergency',
    3: 'Urgent',
    4: 'Less Urgent',
    5: 'Non-Urgent'
}

# KTAS Analysis: Analyze the distribution of KTAS results for experts with KDE plot
plt.figure(figsize=(12, 6))
ax = sns.histplot(data=df, x='KTAS_expert', kde=True, color='gray', bins=10)

plt.title('Distribution of KTAS Results for Experts with KDE Plot')
plt.xlabel('KTAS Result (Experts)')
plt.ylabel('Frequency')

# Setting custom x-axis tick labels with rotation
ax.set_xticks(range(1, 6))
ax.set_xticklabels([ktas_labels[val] for val in range(1, 6)], rotation=45, ha='right')
plt.show()


# In[41]:


# EDA

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming data is in a CSV file named 'your_data.csv'
file_path = '501_project_dataset.tsv'
df = pd.read_csv(file_path, sep="\t")


# Display the first few rows of the DataFrame
print("First few rows of the DataFrame:")
df.head()


# In[42]:


# Summary of the DataFrame, including data types and non-null counts
print("\nSummary of the DataFrame:")
df.info()

# Basic statistics of numerical columns
print("\nBasic statistics of numerical columns:")
df.describe()


# count: This row shows the number of non-missing values for each column. For example, in the "KTAS_expert" column, there are 1267 non-missing values.
# 
# mean: This row represents the mean (average) value for each column. For instance, the mean age in our dataset is approximately 54.42 years.
# 
# std: The standard deviation is a measure of the amount of variation or dispersion in a set of values. A higher standard deviation indicates more spread-out values. In this dataset, the standard deviation for age is approximately 19.73.
# 
# min: The minimum value in each column. For example, the minimum age in our dataset is 16 years.
# 
# 25%: This represents the first quartile (Q1) or the 25th percentile. It indicates the value below which 25% of the data falls. For example, 25% of the ages in this dataset are below 37 years.
# 
# 50%: This is the median or the second quartile (Q2) or the 50th percentile. It represents the middle value of the dataset. In this dataset, the median age is approximately 57 years.
# 
# 75%: This is the third quartile (Q3) or the 75th percentile. It indicates the value below which 75% of the data falls. For example, 75% of the ages in this dataset are below 71 years.
# 
# max: The maximum value in each column. For instance, the maximum age in this dataset is 96 years.

# In[43]:


# Skewness Analysis: Select numeric columns excluding 'Unnamed: 0'
numeric_columns = df.drop(columns=['Unnamed: 0']).select_dtypes(include=['float64', 'int64']).columns

# Calculate skewness for the selected numeric columns
skewness = df[numeric_columns].skew()

# Print a header for the skewness values
print("Skewness:\n")

# Display the calculated skewness values
skewness


# **Skewness it indicates whether the data points in a distribution are concentrated more on one side than the other.**
# 
# Group:Skewness of the 'Group' column. A value around 0 indicates approximately symmetric distribution.
# 
# Sex:Skewness of the 'Sex' column. A value around 0 indicates approximately symmetric distribution.
# 
# Age: Skewness of the 'Age' column. A negative value indicates a longer left tail, meaning the distribution is skewed to the younger age group.
# 
# Patients number per hour: Skewness of the 'Patients number per hour' column. A positive value indicates a longer right tail, meaning the distribution is skewed to higher patient numbers per hour.
# 
# Arrival mode: Skewness of the 'Arrival mode' column. A value around 0 indicates approximately symmetric distribution.
# 
# Injury: Skewness of the 'Injury' column. A positive value indicates a longer right tail, meaning the distribution is skewed towards more instances of injury.
# 
# Mental: Skewness of the 'Mental' column. A positive value indicates a longer right tail, meaning the distribution is skewed towards higher mental state values.
# 
# Pain: Skewness of the 'Pain' column. A negative value indicates a longer left tail, meaning the distribution is skewed towards lower pain values.
# 
# NRS_pain: Skewness of the 'NRS_pain' column. A positive value indicates a longer right tail, meaning the distribution is skewed towards higher Numeric Rating Scale for pain values.
# 
# SBP: Skewness of the 'SBP' (Systolic Blood Pressure) column. A positive value indicates a longer right tail, meaning the distribution is skewed towards higher systolic blood pressure.
# 
# DBP: Skewness of the 'DBP' (Diastolic Blood Pressure) column. A positive value indicates a longer right tail, meaning the distribution is skewed towards higher diastolic blood pressure.
# 
# HR: Skewness of the 'HR' (Heart Rate) column. A positive value indicates a longer right tail, meaning the distribution is skewed towards higher heart rates.
# 
# RR: Skewness of the 'RR' (Respiration Rate) column. A positive value indicates a longer right tail, meaning the distribution is skewed towards higher respiration rates.
# 
# BT: Skewness of the 'BT' (Body Temperature) column. A positive value indicates a longer right tail, meaning the distribution is skewed towards higher body temperatures.
# 
# Saturation: Skewness of the 'Saturation' column. A highly negative value indicates a longer left tail, meaning the distribution is strongly skewed towards lower saturation values. It might be worth investigating the extreme skewness in this variable.
# 
# KTAS_expert: Skewness of the 'KTAS_expert' column. A negative value indicates a longer left tail, meaning the distribution is skewed towards lower KTAS expert ratings.

# In[44]:


# Kurtosis for specific columns
numeric_columns = df.drop(columns=['Unnamed: 0']).select_dtypes(include=['float64', 'int64']).columns
kurtosis_values = df[numeric_columns].kurt()

# Display the kurtosis values
print("Kurtosis:\n")
kurtosis_values


# **Kurtosis is a statistical measure that determines how much a distribution's tails differ from a normal distribution's tails.**
# 
# There are three types of kurtosis:
# 
# Mesokurtic (Normal distribution): indicating that the distribution has similar tails and peak to a normal distribution.
# 
# Leptokurtic (Positive kurtosis):The distribution has heavier tails and a higher peak than a normal distribution. It indicates that the data has more outliers and is more concentrated around the mean.
# 
# Platykurtic (Negative kurtosis): The distribution has lighter tails and a flatter peak than a normal distribution. It suggests that the data has fewer outliers and is more spread out.
# 
# Group, Sex, Age, Patients number per hour, Pain, KTAS_expert: These distributions are relatively flat and have lighter tails compared to a normal distribution (platykurtic).
# 
# Arrival mode, Injury, Mental, NRS_pain, SBP, DBP, HR, RR, BT, Saturation: These distributions have heavier tails and a higher peak compared to a normal distribution (leptokurtic).

# In[45]:


# Define the specific percentiles you want to calculate
percentiles = [0.25, 0.50, 0.75]

# Calculate specific percentiles for numeric columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
custom_percentiles = df[numeric_columns].quantile(percentiles)

# Display the percentiles
print("\nPercentiles:")
custom_percentiles


# **Interpretation of Percentiles:**
# 
# 25th Percentile (Q1):
# Example: The 25th percentile of Age is 37.0, meaning 25% of the individuals are 37 years old or younger.
# 
# 50th Percentile (Median or Q2):
# Example: The median of Age is 57.0, indicating that 50% of the individuals are 57 years old or younger.
# 
# 75th Percentile (Q3):
# Example: The 75th percentile of Age is 71.0, suggesting that 75% of the individuals are 71 years old or younger.
# 
# Example (for Age):
# 25% of the patients are 37 years old or younger.
# 50% (median) of the patients are 57 years old or younger.
# 75% of the patients are 71 years old or younger.

# In[46]:


# Correlation matrix for numeric columns
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df[numeric_columns].corr()

# Display the correlation matrix
print("\nCorrelation Matrix:")
correlation_matrix


# **The correlation matrix, shows the correlation coefficients between different pairs of variables.**
# 
# Correlation Coefficient Range:
# Values range from -1 to 1.
# -1 indicates a perfect negative correlation.
# 1 indicates a perfect positive correlation.
# 0 indicates no correlation.
# 
# Close to 1: Variables move together in the same direction.
# Close to -1: Variables move together in opposite directions.
# Close to 0: Weak or no linear correlation.
# 
# Positive Correlation:
# Age and Patients number per hour: 0.211540 (Weak positive correlation).
# Injury and Mental: 0.000868 (Very weak positive correlation).
# Negative Correlation:
# Pain and NRS_pain: -0.571965 (Moderate negative correlation).
# Pain and DBP: 0.014824 (Very weak negative correlation).
# Strength of Correlation:
# KTAS_expert and Mental: -0.349787 (Moderate negative correlation).
# Saturation and BT: 0.004443 (Very weak positive correlation).
# Missing Values:
# Some correlations involve NaN, indicating missing or undefined values in the dataset.

# In[47]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming your data is in a DataFrame named 'df'

# Check for missing values
missing_values = df.isnull().sum()

# Plot a bar chart to visualize missing values
plt.figure(figsize=(10, 6))
sns.barplot(x=missing_values.index, y=missing_values.values, palette='viridis')
plt.title('Missing Values Distribution')
plt.xlabel('Columns')
plt.ylabel('Number of Missing Values')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[48]:


# initializing the dataframe
import  pandas as pd
df0 = pd.read_csv("501_project_dataset.tsv", sep='\t')  # Reading the data from 'data.tsv' into a pandas DataFrame

df = df0.copy(deep=True)  # Creating a deep copy of the DataFrame
df = df.iloc[:,1:]  # Removing the first column from the DataFrame
df.head()  # Displaying the first few rows of the DataFrame


# In[49]:


# Arrival modes '5' and '7' had very low occurrences (2 each)
# so those rows were dropped to reduce feature size
df = df[df['Arrival mode'] != 5]  # Dropping rows where 'Arrival mode' is equal to 5
df = df[df['Arrival mode'] != 7]  # Dropping rows where 'Arrival mode' is equal to 7

# Replacing NaN values in text features with empty strings
df['Diagnosis in ED'].replace(np.nan, '', inplace=True)  # Replacing NaN values in 'Diagnosis in ED' column with empty strings

df.isnull().sum()  # Checking the number of null values in each column of the DataFrame

# feature-target split
X, y = df.drop(columns=['KTAS_expert'], axis=1), df['KTAS_expert']  # Separating the features (X) and the target variable (y)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=30)  # Splitting the data into training and testing sets

# features where NaN values have to be replaed with 0
feat_impute_0 = ['NRS_pain']
# features where NaN values have to be replaed with mean
feat_impute_mean = ['SBP', 'DBP', 'HR', 'RR', 'BT', 'Saturation']
# features which have to be one-hot encoded
feat_onehot = ['Group', 'Sex', 'Arrival mode', 'Injury', 'Pain']
# rest of the features which only have to be normalized
feat_scale_rest = ['Age', 'Patients number per hour', 'Mental']


# In[50]:


# Creating a SimpleImputer object with strategy as 'constant' and fill value as 0
simple_imputer_0 = SimpleImputer(strategy='constant', fill_value=0)
# Fitting the SimpleImputer on the selected features for imputation
simple_imputer_0.fit(X[feat_impute_0])
# Creating a StandardScaler object for normalizing the imputed values
standard_scaler_impute_0 = StandardScaler()
# Fitting the StandardScaler on the transformed imputed values
standard_scaler_impute_0.fit(simple_imputer_0.transform(X[feat_impute_0]))

# Creating a pipeline for imputation and normalization using the 0 imputer
pipe_impute_0 = Pipeline([
  ('imputer', simple_imputer_0),  # Imputing missing values with 0
  ('scaler', standard_scaler_impute_0)  # Normalizing the imputed values
])

# Creating a SimpleImputer object with default strategy (mean)
simple_imputer_mean = SimpleImputer()
# Fitting the SimpleImputer on the selected features for imputation
simple_imputer_mean.fit(X[feat_impute_mean])
# Creating a StandardScaler object for normalizing the imputed values
standard_scaler_impute_mean = StandardScaler()
# Fitting the StandardScaler on the transformed imputed values
standard_scaler_impute_mean.fit(simple_imputer_mean.transform(X[feat_impute_mean]))

# Creating a pipeline for imputation and normalization using the mean imputer
pipe_impute_mean = Pipeline([
  ('imputer', simple_imputer_mean),  # Imputing missing values with mean
  ('scaler', standard_scaler_impute_mean)  # Normalizing the imputed values
])


# In[51]:


# CountVectorizer for Chief Complaint Translated
count_vectorizer_complain = CountVectorizer()
count_vectorizer_complain.fit(X['Cheif_Complain_Translated'])  # Fitting the CountVectorizer on the 'Cheif_Complain_Translated' column

# TruncatedSVD for Chief Complaint Translated
lsa_complain = TruncatedSVD(n_components=120)
lsa_complain.fit(count_vectorizer_complain.transform(X['Cheif_Complain_Translated']))  # Fitting the TruncatedSVD on the transformed data from CountVectorizer

# text feature vectorizer and decomposition pipeline
# Pipeline for Chief Complaint Translated
pipe_complain = Pipeline([
  ('cvec', count_vectorizer_complain),  # CountVectorizer step in the pipeline
  ('lsa', lsa_complain)  # TruncatedSVD step in the pipeline
])


# CountVectorizer for Diagnosis in ED
count_vectorizer_diagnosis = CountVectorizer()
count_vectorizer_diagnosis.fit(X['Diagnosis in ED'])  # Fitting the CountVectorizer on the 'Diagnosis in ED' column

# TruncatedSVD for Diagnosis in ED
lsa_diagnosis = TruncatedSVD(n_components=120)
lsa_diagnosis.fit(count_vectorizer_diagnosis.transform(X['Diagnosis in ED']))  # Fitting the TruncatedSVD on the transformed data from CountVectorizer

# text feature vectorizer and decomposition pipeline
# Pipeline for Diagnosis in ED
pipe_diagnosis = Pipeline([
  ('cvec', count_vectorizer_diagnosis),  # CountVectorizer step in the pipeline
  ('lsa', lsa_diagnosis)  # TruncatedSVD step in the pipeline
])


# In[52]:


# one-hot encoder
one_hot_encoder =  OneHotEncoder(drop='first') # drop first to avoid dummy variable trap
one_hot_encoder.fit(X[feat_onehot]) # fit the one-hot encoder on the selected features

# standard normalizing saler
standard_scaler_rest = StandardScaler() # standard scaler for the rest of the features
standard_scaler_rest.fit(X[feat_scale_rest]) # fit the standard scaler on the selected features

# Column transformer to conatenate the various pipelines
pre_proc_cols = ColumnTransformer([
    ('impute_0', pipe_impute_0, feat_impute_0),
    ('impute_mean', pipe_impute_mean, feat_impute_mean),
    ('cv_complains', pipe_complain, 'Cheif_Complain_Translated'),
    ('cv_diagnosis', pipe_diagnosis, 'Diagnosis in ED'),
    ('one_hot', one_hot_encoder, feat_onehot),
    ('scale_rest', standard_scaler_rest, feat_scale_rest)
])


# In[53]:


# Traige Accuracy metric.
def triage_accuracy(y, y_hat):
  tru = 0
  for tr, pr in zip(y, y_hat):
    # the accuracy window is tunable, currently set for maximizing surety in exchange for increased window size
    if (pr - tr < 1.6 and pr - tr >= 0) or (tr - pr < 1.2 and tr - pr >= 0):
      tru+=1
  return tru/len(y)

from sklearn.metrics import make_scorer # to make a custom scoring function
triage_score = make_scorer(triage_accuracy) # triage accuracy score for grid search

# helper function to map model's prediction to output class
def y_class(y):
  if y < 1.4:
    return 1
  elif y < 2.4:
    return 2
  elif y < 3.4:
    return 3
  elif y < 4.4:
    return 4
  else:
    return 5


# In[54]:


# function to map model's prediction to output class
def triage_output(y_hat):
  y_out = [y_class(y) for y in y_hat] # map model's prediction to output class
  return np.array(y_out) # return the output class as a numpy array


# a pseudo precision-recall-f1 score metric
# Importing the required library
import collections


# Defining a function to calculate precision, recall, and F1-score for each class
def triage_precision_recall_f1(y, y_hat):

  # Counting the true labels
  true_count_by_label = collections.Counter(y)

  # Converting the model's predictions to output class labels
  y_out = triage_output(y_hat)

  # Counting the predicted labels
  pred_count_by_label = collections.Counter(y_out)

  # Initializing a dictionary to store the number of correct predictions for each label
  correct_by_label = {
      1:0, 2:0, 3:0, 4:0, 5:0
  }

  # Calculating the number of correct predictions for each label
  for tr, pr in zip(y, y_hat):
    if ( (tr + 0.6) > pr and (tr - 0.4) < pr ) :
      correct_by_label[tr] += 1
    elif ( (tr + 1.2) > pr and (tr - 0.8) < pr ) :
      correct_by_label[tr] += 0.4
    elif ( (tr + 1.5) > pr and (tr - 1) < pr ) :
      correct_by_label[tr] += 0.2
    elif ( pr > 5 and tr == 5 ):
      correct_by_label[5] += 1
    elif ( pr < 1 and tr == 1 ):
      correct_by_label[1] += 1

  precision = {}
  recall = {}
  f1 = {}

  # Calculating precision for each label
  for label in range(1, 6):
    if pred_count_by_label[label] != 0:
      precision[label] = correct_by_label[label] / pred_count_by_label[label]
    else:
      precision[label] = 0.0

  # Calculating recall for each label
  for label in range(1, 6):
    if true_count_by_label[label] != 0:
      recall[label] = correct_by_label[label] / true_count_by_label[label]
    else:
      recall[label] = 0.0

  # Calculating F1-score for each label
  for label in range(1, 6):
    if precision[label] + recall[label] != 0:
      f1[label] = (2 * precision[label] * recall[label]) / (precision[label] + recall[label])
    else:
      f1[label] = 0.0

  # Calculating the mean precision, recall, and F1-score
  mean_val = {
      'mean precision': np.array(list(precision.values())).mean(),
      'mean recall': np.array(list(recall.values())).mean(),
      'mean f1': np.array(list(f1.values())).mean(),
  }

  # Returning the precision, recall, F1-score, and mean values
  return [precision, recall,  f1, mean_val]


# to display the pipeline as a diagram
set_config(display="diagram")


# In[55]:


# Defining a function to print various metrics
def print_metrics(prec_recall_f1):
  print('Classwise weak precision:')
  print(prec_recall_f1[0])
  print('Classwise weak recall:')
  print(prec_recall_f1[1])
  print('Classwise weak f1-score:')
  print(prec_recall_f1[2])
  print('Mean weak precision:')
  print(prec_recall_f1[3]['mean precision'])
  print('Mean weak recall:')
  print(prec_recall_f1[3]['mean recall'])
  print('Mean weak f1-score:')
  print(prec_recall_f1[3]['mean f1'])

# Defining a function to print grid search results
def print_grid(grid, X_test, y_test):
  print('Best Score:')
  print(grid.best_score_)
  print('Best Params:')
  print(grid.best_params_)
  print('Accuracy on test set:')
  print(triage_score(grid, X_test, y_test))


# In[56]:


#------------------
# Training the pipeline using various regression methods.
# Linear Regression
# Importing the LinearRegression class from the sklearn.linear_model module
from sklearn.linear_model import LinearRegression

# Creating an instance of the LinearRegression class
lin = LinearRegression()

# Creating a pipeline with preprocessing steps and the LinearRegression classifier
pipe_lin = Pipeline([
  ('prep_proc', pre_proc_cols),  # Preprocessing step
  ('clf', lin)  # LinearRegression classifier
])

# Creating an empty dictionary for the parameters of grid search
lin_params = {}

# Creating a GridSearchCV object with the pipeline, parameters, and cross-validation
lin_grid = GridSearchCV(pipe_lin, lin_params, cv=10, scoring=triage_score)

# Fitting the GridSearchCV object to the training data
lin_grid.fit(X_train, y_train)

# Predicting the target variable using the fitted model
y_pred = lin_grid.predict(X_test)

# Printing the grid search results and evaluation metrics
print_grid(lin_grid, X_test, y_test)

# Calculating precision, recall, and F1-score
prec_recall_f1 = triage_precision_recall_f1(y_test, y_pred)

# Printing the evaluation metrics
print_metrics(prec_recall_f1)

pipe_lin


# In[57]:


# Ridge Regression
# Importing the Ridge class from the sklearn.linear_model module
from sklearn.linear_model import Ridge

# Creating an instance of the Ridge class
rig = Ridge()

# Creating a pipeline with preprocessing steps and the Ridge classifier
pipe_rig = Pipeline([
  ('prep_proc', pre_proc_cols),  # Preprocessing step
  ('clf', rig)  # Ridge classifier
])

# Creating an empty dictionary for the parameters of grid search
rig_params = {}

# Setting the alpha parameter values for Ridge regularization
rig_params['clf__alpha'] = [0.001, 0.01, 0.1, 1.0, 10]

# Creating an instance of GridSearchCV with the Ridge pipeline and parameters
rig_grid = GridSearchCV(pipe_rig, rig_params, cv=10, scoring=triage_score)

# Fitting the Ridge grid search on the training data
rig_grid.fit(X_train, y_train)

# Creating a DataFrame with the Ridge grid search results
rig_res = pd.DataFrame(rig_grid.cv_results_)[['param_clf__alpha', 'mean_test_score']]

# Saving the Ridge grid search results to a CSV file
rig_res.to_csv('ridge_params_vs_score.csv')

# Printing the grid search results and accuracy on the test set
print_grid(rig_grid, X_test, y_test)

# Predicting the target variable using the Ridge grid search model
y_pred = rig_grid.predict(X_test)

# Calculating precision, recall, and F1 score for the predicted values
prec_recall_f1 = triage_precision_recall_f1(y_test, y_pred)

# Printing the precision, recall, and F1 score metrics
print_metrics(prec_recall_f1)


# In[58]:


# Lasso Regression
# Importing the Lasso class from the sklearn.linear_model module
from sklearn.linear_model import Lasso

# Creating an instance of the Lasso class
las = Lasso()

# Creating a pipeline with preprocessing steps and the Lasso classifier
pipe_las = Pipeline([
  ('prep_proc', pre_proc_cols),  # Preprocessing step
  ('clf', las)  # Lasso classifier
])

# Creating an empty dictionary for the parameters of grid search
las_params = {}

# Setting the alpha parameter values for Lasso regularization
las_params['clf__alpha'] = [0.001, 0.01, 0.1, 1.0, 10]

# Creating a GridSearchCV object with the pipeline, parameters, and 10-fold cross-validation
las_grid = GridSearchCV(pipe_las, las_params, cv=10, scoring=triage_score)

# Fitting the GridSearchCV object to the training data
las_grid.fit(X_train, y_train)

# Creating a DataFrame with the Lasso grid search results
las_res = pd.DataFrame(las_grid.cv_results_)[['param_clf__alpha', 'mean_test_score']]

# Saving the Lasso grid search results to a CSV file
las_res.to_csv('lasso_params_vs_score.csv')

# Printing the evaluation metrics for the Lasso grid search
print_grid(las_grid, X_test, y_test)

# Predicting the target variable using the Lasso grid search model
y_pred = las_grid.predict(X_test)

# Calculating the precision, recall, and F1-score for the Lasso predictions
prec_recall_f1 = triage_precision_recall_f1(y_test, y_pred)

# Printing the evaluation metrics for the Lasso predictions
print_metrics(prec_recall_f1)


# In[59]:


# Stochastic Gradient Descent (SGD) Classifier
# Importing the SGDClassifier class from the sklearn.linear_model module
from sklearn.linear_model import SGDClassifier

# Creating an instance of the SGDClassifier class with max_iter set to 20000
sgd = SGDClassifier(max_iter=20000)

# Creating a pipeline with preprocessing steps and the SGDClassifier
pipe_sgd = Pipeline([
  ('prep_proc', pre_proc_cols),  # Preprocessing step
  ('clf', sgd)  # SGDClassifier
])

# Creating an empty dictionary for the parameters of grid search
sgd_params = {}

# Creating a GridSearchCV object with the pipeline, parameters, and 10-fold cross-validation
sgd_grid = GridSearchCV(pipe_sgd, sgd_params, cv=10, scoring=triage_score)

# Fitting the grid search on the training data
sgd_grid.fit(X_train, y_train)

# Printing the grid search results on the test data
print_grid(sgd_grid, X_test, y_test)

# Predicting the target variable on the test data using the best estimator from grid search
y_pred = sgd_grid.predict(X_test)

# Calculating precision, recall, and F1-score using the predicted and actual target values
prec_recall_f1 = triage_precision_recall_f1(y_test, y_pred)

# Printing the evaluation metrics
print_metrics(prec_recall_f1)


# In[60]:


# Support Vector Regression (SVR)
# Importing the SVR class from the sklearn.svm module
from sklearn.svm import SVR

# Creating an instance of the SVR class
svr = SVR()

# Creating a pipeline with preprocessing steps and the SVR classifier
pipe_svr = Pipeline([
  ('prep_proc', pre_proc_cols),  # Preprocessing step
  ('clf', svr)  # SVR classifier
])

# Defining the parameters for grid search
svr_params = {}
svr_params['clf__kernel'] = ['poly', 'rbf', 'sigmoid']  # Different kernel functions
svr_params['clf__C'] = [0.01, 0.1, 1.0, 10, 100]  # Different values for the regularization parameter C

# Creating a grid search object with the pipeline and parameters
svr_grid = GridSearchCV(pipe_svr, svr_params, cv=10, scoring=triage_score)

# Fitting the grid search object to the training data
svr_grid.fit(X_train, y_train)

# Creating a DataFrame with the grid search results
svr_res = pd.DataFrame(svr_grid.cv_results_)[['param_clf__C', 'param_clf__kernel', 'mean_test_score']]

# Saving the grid search results to a CSV file
svr_res.to_csv('SVR_params_vs_score.csv')

# Printing the grid search results
print_grid(svr_grid, X_test, y_test)

# Predicting the target variable using the trained model
y_pred = svr_grid.predict(X_test)

# Calculating precision, recall, and F1 score
prec_recall_f1 = triage_precision_recall_f1(y_test, y_pred)

# Printing the evaluation metrics
print_metrics(prec_recall_f1)


# In[61]:


# K-Nearest Neighbors Regressor (KNN)
# Importing the KNeighborsRegressor class from the sklearn.neighbors module
from sklearn.neighbors import KNeighborsRegressor

# Creating an instance of KNeighborsRegressor
knn = KNeighborsRegressor()

# Creating a pipeline with pre_proc_cols preprocessing steps and knn as the regressor
pipe_knn = Pipeline([
  ('prep_proc', pre_proc_cols),
  ('clf', knn)
])

# Defining an empty dictionary for knn_params
knn_params = {}

# Setting the 'n_neighbors' parameter values for KNeighborsRegressor
knn_params['clf__n_neighbors'] = [3, 5, 7]

# Setting the 'weights' parameter values for KNeighborsRegressor
knn_params['clf__weights'] = ['uniform', 'distance']

# Creating a grid search object with the pipeline and parameters
knn_grid = GridSearchCV(pipe_knn, knn_params, cv=10, scoring=triage_score)

# Fitting the grid search object to the training data
knn_grid.fit(X_train, y_train)

# Creating a DataFrame with the grid search results
knn_res = pd.DataFrame(knn_grid.cv_results_)[['param_clf__n_neighbors', 'param_clf__weights', 'mean_test_score']]

# Saving the grid search results to a CSV file
knn_res.to_csv('knn_params_vs_score.csv')

# Printing the grid search results
print_grid(knn_grid, X_test, y_test)

# Predicting the target variable using the trained model
y_pred = knn_grid.predict(X_test)

# Calculating precision, recall, and F1 score
prec_recall_f1 = triage_precision_recall_f1(y_test, y_pred)

# Printing the evaluation metrics
print_metrics(prec_recall_f1)


# In[62]:


# Partial Least Squares Regression (PLS)
# Importing the PLSRegression class from the sklearn.cross_decomposition module
from sklearn.cross_decomposition import PLSRegression

# Creating an instance of PLSRegression
pls = PLSRegression()

# Creating a pipeline with pre_proc_cols preprocessing steps and pls as the regressor
pipe_pls = Pipeline([
  ('prep_proc', pre_proc_cols),
  ('clf', pls)
])

# Defining an empty dictionary for pls_params
pls_params = {}

# Setting the 'n_components' parameter values for PLSRegression
pls_params['clf__n_components'] = [1, 2, 5, 10, 15]

# Creating a GridSearchCV object with pipe_pls as the estimator, pls_params as the parameter grid,
# 10-fold cross-validation, and triage_score as the scoring metric
pls_grid = GridSearchCV(pipe_pls, pls_params, cv=10, scoring=triage_score)

# Fitting the PLSRegression model to the training data
pls_grid.fit(X_train, y_train)

# Printing the grid search results for PLSRegression on the test data
print_grid(pls_grid, X_test, y_test)

# Predicting the target variable using the fitted PLSRegression model on the test data
y_pred = pls_grid.predict(X_test)

# Calculating precision, recall, and F1-score for the predicted values
prec_recall_f1 = triage_precision_recall_f1(y_test, y_pred)

# Printing the evaluation metrics (precision, recall, and F1-score)
print_metrics(prec_recall_f1)


# In[63]:


# Bayesian Ridge Regression
# Importing the BayesianRidge class from the sklearn.linear_model module
from sklearn.linear_model import BayesianRidge

# Creating an instance of BayesianRidge
brg = BayesianRidge()

# Creating a pipeline with pre_proc_cols preprocessing steps and brg as the regressor
pipe_brg = Pipeline([
  ('prep_proc', pre_proc_cols),
  ('clf', brg)
])

# Defining an empty dictionary for brg_params
brg_params = {}

# Creating a GridSearchCV object with pipe_brg as the estimator, brg_params as the parameter grid,
# 10-fold cross-validation, and triage_score as the scoring metric
brg_grid = GridSearchCV(pipe_brg, brg_params, cv=10, scoring=triage_score)

# Fitting the brg_grid object to the training data X_train and y_train
brg_grid.fit(X_train, y_train)

# Printing the grid search results using the print_grid function for the brg_grid object and the test data X_test and y_test
print_grid(brg_grid, X_test, y_test)

# Predicting the target variable y_pred using the brg_grid object and the test data X_test
y_pred = brg_grid.predict(X_test)

# Calculating precision, recall, and F1 score using the triage_precision_recall_f1 function for the y_test and y_pred
prec_recall_f1 = triage_precision_recall_f1(y_test, y_pred)

# Printing the evaluation metrics using the print_metrics function for the prec_recall_f1 values
print_metrics(prec_recall_f1)


# In[64]:


# Decision Tree Regressor
# Importing the DecisionTreeRegressor class from the sklearn.tree module
from sklearn.tree import DecisionTreeRegressor

# Creating an instance of DecisionTreeRegressor
dtr = DecisionTreeRegressor()

# Creating a pipeline with pre_proc_cols preprocessing steps and dtr as the regressor
pipe_dtr = Pipeline([
  ('prep_proc', pre_proc_cols),
  ('clf', dtr)
])

# Defining the parameters for grid search
dtr_params = {}
dtr_params['clf__criterion'] = ['squared_error', 'friedman_mse']
dtr_params['clf__splitter'] = ['best', 'random']

# Creating a GridSearchCV object with pipe_dtr, dtr_params, cv=10, and scoring=triage_score
dtr_grid = GridSearchCV(pipe_dtr, dtr_params, cv=10, scoring=triage_score)

# Fitting the grid search on X_train and y_train
dtr_grid.fit(X_train, y_train)

# Creating a DataFrame with the grid search results for 'param_clf__splitter', 'param_clf__criterion', and 'mean_test_score'
dtr_res = pd.DataFrame(dtr_grid.cv_results_)[['param_clf__splitter', 'param_clf__criterion', 'mean_test_score']]

# Saving the DataFrame as a CSV file named 'dtr_params_vs_score.csv'
dtr_res.to_csv('dtr_params_vs_score.csv')

# Printing the grid search results using the print_grid function on X_test and y_test
print_grid(dtr_grid, X_test, y_test)

# Predicting the target variable using the fitted grid search model on X_test
y_pred = dtr_grid.predict(X_test)

# Calculating precision, recall, and F1 score using triage_precision_recall_f1 function on y_test and y_pred
prec_recall_f1 = triage_precision_recall_f1(y_test, y_pred)

# Printing the evaluation metrics using the print_metrics function on prec_recall_f1
print_metrics(prec_recall_f1)


# In[65]:


# Random Forest Regressor
# Importing the RandomForestRegressor class from the sklearn.ensemble module
from sklearn.ensemble import RandomForestRegressor

# Creating an instance of RandomForestRegressor
rft = RandomForestRegressor()

# Creating a pipeline with pre_proc_cols preprocessing steps and rft as the regressor
pipe_rft = Pipeline([
  ('prep_proc', pre_proc_cols),
  ('clf', rft)
])

# Defining the parameters for grid search
rft_params = {}
rft_params['clf__n_estimators'] = [20, 50, 80, 100, 150, 200]
rft_params['clf__max_depth'] = [None, 3, 10, 20, 30]

# Creating an instance of GridSearchCV with pipe_rft, rft_params, cv=10, and scoring=triage_score
rft_grid = GridSearchCV(pipe_rft, rft_params, cv=10, scoring=triage_score)

# Fitting the rft_grid on X_train and y_train
rft_grid.fit(X_train, y_train)

# Creating a DataFrame with the results of the grid search
rft_res = pd.DataFrame(rft_grid.cv_results_)[['param_clf__n_estimators',
                                              'param_clf__max_depth',
                                              'mean_test_score']]

# Saving the DataFrame to a CSV file
rft_res.to_csv('rft_params_vs_score.csv')

# Printing the grid search results
print_grid(rft_grid, X_test, y_test)

# Predicting the target variable using the trained model
y_pred = rft_grid.predict(X_test)


# In[66]:


# AdaBoost Regressor
# Importing the AdaBoostRegressor class from the sklearn.ensemble module
from sklearn.ensemble import AdaBoostRegressor
# Importing the DecisionTreeRegressor class from the sklearn.tree module
from sklearn.tree import DecisionTreeRegressor  # for base estimator

# Creating an instance of AdaBoostRegressor
ada = AdaBoostRegressor()

# Creating a pipeline with pre_proc_cols preprocessing steps and ada as the regressor
pipe_ada = Pipeline([
  ('prep_proc', pre_proc_cols),
  ('clf', ada)
])

# Defining the parameters for grid search
ada_params = {}
ada_params['clf__n_estimators'] = [50, 100, 150, 200]
ada_params['clf__base_estimator'] = [DecisionTreeRegressor(max_depth=3),
                                     DecisionTreeRegressor(max_depth=10),
                                     DecisionTreeRegressor(max_depth=20)]

# Creating an instance of GridSearchCV with pipe_ada, ada_params, cv=10, and scoring=triage_score
ada_grid = GridSearchCV(pipe_ada, ada_params, cv=10, scoring=triage_score)
# Fitting the GridSearchCV object to X_train and y_train
ada_grid.fit(X_train, y_train)

# Creating a DataFrame from ada_grid.cv_results_ with selected columns and saving it to a CSV file
ada_res = pd.DataFrame(ada_grid.cv_results_)[['param_clf__n_estimators',
                                              'param_clf__base_estimator',
                                              'mean_test_score']]
ada_res.to_csv('ada_params_vs_score.csv')

# Printing the grid search results using the print_grid function
print_grid(ada_grid, X_test, y_test)

# Predicting the target variable using ada_grid and X_test
y_pred = ada_grid.predict(X_test)

# Calculating precision, recall, and F1 score using triage_precision_recall_f1 function
prec_recall_f1 = triage_precision_recall_f1(y_test, y_pred)
# Printing the metrics using the print_metrics function
print_metrics(prec_recall_f1)


# In[67]:


# Gradient Boosting Regressor
# Importing the GradientBoostingRegressor class from the sklearn.ensemble module
from sklearn.ensemble import GradientBoostingRegressor

# Creating an instance of GradientBoostingRegressor
gdb = GradientBoostingRegressor()

# Creating a pipeline with pre_proc_cols preprocessing steps and gdb as the regressor
pipe_gdb = Pipeline([
  ('prep_proc', pre_proc_cols),
  ('clf', gdb)
])

# Defining the parameters for grid search
gdb_params = {}
gdb_params['clf__learning_rate'] = [0.01, 0.1, 1]
gdb_params['clf__n_estimators'] = [50, 100, 150, 200]

# Creating a GridSearchCV object with pipe_gdb as the estimator, gdb_params as the parameter grid,
# cv=10 for 10-fold cross-validation, and triage_score as the scoring metric
gdb_grid = GridSearchCV(pipe_gdb, gdb_params, cv=10, scoring=triage_score)
gdb_grid.fit(X_train, y_train)

# Creating a DataFrame with the grid search results
gdb_res = pd.DataFrame(gdb_grid.cv_results_)[['param_clf__n_estimators',
                                              'param_clf__learning_rate',
                                              'mean_test_score']]
gdb_res.to_csv('gdb_params_vs_score.csv')

# Printing the grid search results
print_grid(gdb_grid, X_test, y_test)

# Predicting the target variable using the best estimator from grid search
y_pred = gdb_grid.predict(X_test)

# Calculating precision, recall, and F1 score
prec_recall_f1 = triage_precision_recall_f1(y_test, y_pred)

# Printing the evaluation metrics
print_metrics(prec_recall_f1)


# In[68]:


# Multi-layer Perceptron Regressor (MLP)
# Importing the MLPRegressor class from the sklearn.neural_network module
from sklearn.neural_network import MLPRegressor

# Creating an instance of MLPRegressor with max_iter=1000
mlp = MLPRegressor(max_iter=1000)

# Creating a pipeline with pre_proc_cols preprocessing steps and mlp as the classifier
pipe_mlp = Pipeline([
  ('prep_proc', pre_proc_cols),
  ('clf', mlp)
])

# Defining the parameters for grid search
mlp_params = {}
mlp_params['clf__hidden_layer_sizes'] = [(10,10), (20,20), (20,10,10)]

# Creating a GridSearchCV object with pipe_mlp as the estimator, mlp_params as the parameter grid,
# cv=10 for 10-fold cross-validation, and triage_score as the scoring metric
mlp_grid = GridSearchCV(pipe_mlp, mlp_params, cv=10, scoring=triage_score)

# Fitting the grid search object to the training data
mlp_grid.fit(X_train, y_train)

# Creating a DataFrame with the grid search results for hidden_layer_sizes and mean_test_score
mlp_res = pd.DataFrame(mlp_grid.cv_results_)[['param_clf__hidden_layer_sizes', 'mean_test_score']]

# Saving the DataFrame as a CSV file
mlp_res.to_csv('mlp_params_vs_score.csv')

# Printing the grid search results for the best estimator, X_test, and y_test
print_grid(mlp_grid, X_test, y_test)

# Predicting the target variable for X_test using the best estimator from the grid search
y_pred = mlp_grid.predict(X_test)

# Calculating precision, recall, and F1 score for y_test and y_pred
prec_recall_f1 = triage_precision_recall_f1(y_test, y_pred)

# Printing the precision, recall, and F1 score metrics
print_metrics(prec_recall_f1)


# In[69]:


pipe_mlp


# In[ ]:




