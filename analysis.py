import pandas as pd
import numpy as np
from datetime import datetime
import time
from sklearn.preprocessing import OneHotEncoder

# Debug: Confirm code version
print("Running code updated at 04:35 PM BST, July 14, 2025")

# Set pandas display options to show all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Load the dataset
df = pd.read_csv('C:/Users/benji/Desktop/BenProjects/house-price-analysis/train.csv')

# Print first 5 rows
print(df.head())

# Print missing values per column
print("Missing values per column:")
print(df.isnull().sum())

# Save missing values to a file for review
with open('missing_values.txt', 'w') as f:
    f.write(df.isnull().sum().to_string())
print("Missing values saved to missing_values.txt")

# Print data types of columns
print("\nData types:")
print(df.dtypes)

# Print basic statistics
print("\nShape:", df.shape)
print("\nSalePrice stats:")
print(df['SalePrice'].describe())

# Initial cleaning
# Handle missing values with context-specific imputation
for column in df.columns:
    if column == 'LotFrontage':  # Impute LotFrontage by median of Neighborhood
        df[column] = df.groupby('Neighborhood')[column].transform(lambda x: x.fillna(x.median()))
    elif df[column].dtype in ['int64', 'float64']:
        df[column] = df[column].fillna(df[column].median())
    else:
        df[column] = df[column].fillna("None")  # Use "None" for categorical

# Remove outliers (based on IQR for SalePrice)
Q1 = df['SalePrice'].quantile(0.25)
Q3 = df['SalePrice'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['SalePrice'] >= max(10000, lower_bound)) & (df['SalePrice'] <= upper_bound)]

# Encode categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
encoded_cols = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))
df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

# Print cleaned shape and missing values
print("\nShape after cleaning and encoding:", df.shape)
print("Missing values after cleaning:\n", df.isnull().sum())

# Save cleaned dataframe to CSV
df.to_csv('cleaned_houses.csv', index=False)
print("Cleaned data saved as cleaned_houses.csv")