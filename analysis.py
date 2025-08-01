import pandas as pd
import numpy as np
from datetime import datetime
import time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import sqlite3
import joblib



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

# Create interaction and squared feature
df['OverallQual_Remod'] = df['OverallQual'] * df['YearRemodAdd']
df['OverallQual_Squared'] = df['OverallQual'] ** 2

# Select features and target
features = ['LotArea', 'YearBuilt', 'YearRemodAdd', 'OverallQual', 'OverallCond', 'OverallQual_Remod', 'OverallQual_Squared', 
            'TotalBsmtSF', 'GrLivArea', '1stFlrSF', '2ndFlrSF', 'GarageArea', 'BedroomAbvGr', 'FullBath', 'TotRmsAbvGrd', 
            'GarageCars', 'Fireplaces', 'GarageYrBlt', 'MasVnrArea', 'KitchenQual_Gd', 'Neighborhood_CollgCr', 
            'Neighborhood_NoRidge', 'Exterior1st_VinylSd']
X = df[features]
y = np.log1p(df['SalePrice'])  # Log transform target for better modeling

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Gradient Boosting model
model = GradientBoostingRegressor(n_estimators=75, max_depth=120, learning_rate=0.05, min_samples_leaf=10, random_state=42)
start_time = time.time()
model.fit(X_train_scaled, y_train)
print(f"Training completed in {time.time() - start_time:.2f} seconds")

# Cross-validation check
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-validation MSE scores: {[-s for s in cv_scores]}")
print(f"Mean CV MSE: {np.mean([-s for s in cv_scores]):.2f}")

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

# Save the model
joblib.dump(model, 'house_price_model.joblib')
print("Model saved as house_price_model.joblib")

# SQLite setup with single connection
conn = sqlite3.connect('houses.db')
cursor = conn.cursor()

# Create houses table with updated columns
cursor.execute('''
    CREATE TABLE IF NOT EXISTS houses (
        id INTEGER PRIMARY KEY,
        lot_area REAL,
        year_built INTEGER,
        year_remod_add INTEGER,
        overall_qual INTEGER,
        overall_cond INTEGER,
        overall_qual_remod REAL,
        overall_qual_squared REAL,
        total_bsmt_sf REAL,
        gr_liv_area REAL,
        first_flr_sf REAL,
        second_flr_sf REAL,
        garage_area REAL,
        bedroom_abv_gr INTEGER,
        full_bath INTEGER,
        tot_rms_abv_grd INTEGER,
        garage_cars INTEGER,
        fireplaces INTEGER,
        garage_yr_blt REAL,
        mas_vnr_area REAL,
        sale_price REAL,
        predicted_revenue REAL DEFAULT NULL
    )
''')

# Load cleaned data and insert into table with predicted_revenue as NULL
df_clean = df[['LotArea', 'YearBuilt', 'YearRemodAdd', 'OverallQual', 'OverallCond', 'TotalBsmtSF', 'GrLivArea', 
               '1stFlrSF', '2ndFlrSF', 'GarageArea', 'BedroomAbvGr', 'FullBath', 'TotRmsAbvGrd', 'GarageCars', 
               'Fireplaces', 'GarageYrBlt', 'MasVnrArea', 'SalePrice']].copy()
df_clean['OverallQual_Remod'] = df['OverallQual'] * df['YearRemodAdd']
df_clean['OverallQual_Squared'] = df['OverallQual'] ** 2
df_clean = df_clean.rename(columns={'SalePrice': 'sale_price', '1stFlrSF': 'first_flr_sf', '2ndFlrSF': 'second_flr_sf', 
                                   'OverallQual_Remod': 'overall_qual_remod', 'OverallQual_Squared': 'overall_qual_squared', 
                                   'GarageYrBlt': 'garage_yr_blt', 'MasVnrArea': 'mas_vnr_area'})
df_clean['predicted_revenue'] = None
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

# Generate a basic report
with open('report.md', 'w') as f:
    f.write("# House Price Prediction Report\n")
    f.write("## Introduction\n")
    f.write("This project predicts house sale prices using the Kaggle House Prices dataset.\n")
    f.write("## Data Cleaning\n")
    f.write("- Handled missing values (e.g., LotFrontage imputed by Neighborhood median).\n")
    f.write("- Removed outliers using IQR on SalePrice.\n")
    f.write("## Modeling\n")
    f.write("- Used Gradient Boosting Regressor with max_depth=100, n_estimators=75, learning_rate=0.05.\n")
    f.write("- Features: LotArea, YearBuilt, YearRemodAdd, OverallQual, OverallCond, OverallQual_Remod, OverallQual_Squared, etc.\n")
    f.write("## Results\n")
    f.write(f"- Max Predicted Value: ${np.max(y_pred_original):,.2f}\n")
    f.write(f"- Mean Squared Error (original scale): ${mse_original:,.2f}\n")
    f.write("- Sample Predictions: See predictions_houses.csv\n")
    f.write("## Limitations\n")
    f.write("The model fits the dataset range ($755,000 max) and struggles to extrapolate to $1M without additional high-value data.\n")
    f.write("## Future Work\n")
    f.write("- Collect data with houses > $1M for better extrapolation.\n")
    f.write("- Explore advanced models (e.g., XGBoost, neural networks).\n")
print("Report saved as report.md")