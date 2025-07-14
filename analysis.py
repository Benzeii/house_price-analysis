import pandas as pd
import numpy as np
from datetime import datetime
import time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import sqlite3

# Debug: Confirm code version
print("Running code updated at 05:15 PM BST, July 14, 2025")

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
for column in df.columns:
    if column == 'LotFrontage':
        df[column] = df.groupby('Neighborhood')[column].transform(lambda x: x.fillna(x.median()))
    elif df[column].dtype in ['int64', 'float64']:
        df[column] = df[column].fillna(df[column].median())
    else:
        df[column] = df[column].fillna("None")

# Remove outliers (based on IQR for SalePrice)
Q1 = df['SalePrice'].quantile(0.25)
Q3 = df['SalePrice'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['SalePrice'] >= max(10000, lower_bound)) & (df['SalePrice'] <= upper_bound)].reset_index(drop=True)

# Encode categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
encoded_cols = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)
df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

# Select features and target
features = ['LotArea', 'YearBuilt', 'OverallQual', 'TotalBsmtSF', 'GrLivArea', 'GarageArea', 'Neighborhood_CollgCr', 
            'Neighborhood_NoRidge', 'Exterior1st_VinylSd', 'KitchenQual_Gd']
X = df[features]
y = np.log1p(df['SalePrice'])  # Log transform target for better modeling

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Gradient Boosting model
model = GradientBoostingRegressor(n_estimators=50, max_depth=30, learning_rate=0.1, random_state=42)
start_time = time.time()
model.fit(X_train_scaled, y_train)
print(f"Training completed in {time.time() - start_time:.2f} seconds")

# Predict on test set
y_pred_log = model.predict(X_test_scaled)

# Reverse log transformation with clipping
y_pred_original = np.expm1(y_pred_log)
y_test_original = np.expm1(y_test)
print(f"Max predicted value before clipping: {np.max(y_pred_original):.2e}")
y_pred_original = np.clip(y_pred_original, 1000, 1e6)  # Target range up to $1M
print(f"Max predicted value after clipping: {np.max(y_pred_original):.2e}")

# Calculate mean squared error
mse_log = mean_squared_error(y_test, y_pred_log)
mse_original = mean_squared_error(y_test_original, y_pred_original)
print("Mean Squared Error (log scale):", mse_log)
print("Mean Squared Error (original scale):", mse_original)

# Save predictions to dataframe with titles
df_test = pd.DataFrame(X_test_scaled, columns=features, index=X_test.index)
df_test['actual_revenue'] = y_test_original
df_test['predicted_revenue'] = y_pred_original
df_test.to_csv('predictions_houses.csv', index_label='id')
print("Sample predictions saved to predictions_houses.csv")

# SQLite setup with single connection
conn = sqlite3.connect('houses.db')
cursor = conn.cursor()

# Create houses table with updated columns
cursor.execute('''
    CREATE TABLE IF NOT EXISTS houses (
        id INTEGER PRIMARY KEY,
        lot_area REAL,
        year_built INTEGER,
        overall_qual INTEGER,
        total_bsmt_sf REAL,
        gr_liv_area REAL,
        garage_area REAL,
        sale_price REAL,
        predicted_revenue REAL DEFAULT NULL
    )
''')

# Load cleaned data and insert into table
df_clean = df[['LotArea', 'YearBuilt', 'OverallQual', 'TotalBsmtSF', 'GrLivArea', 'GarageArea', 'SalePrice']].copy()
df_clean.index.name = 'id'
df_clean.to_sql('houses', conn, if_exists='replace', index=True)

# Update with predictions
for idx, row in df_test.iterrows():
    cursor.execute('UPDATE houses SET predicted_revenue = ? WHERE id = ?', (row['predicted_revenue'], idx))

# Verify
cursor.execute('SELECT id, sale_price, predicted_revenue FROM houses WHERE predicted_revenue IS NOT NULL LIMIT 5')
rows = cursor.fetchall()
print("Updated rows with predictions:")
for row in rows:
    print(row)

# Close connection
conn.commit()
conn.close()