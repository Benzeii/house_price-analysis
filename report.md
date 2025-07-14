# House Price Prediction Report
## Introduction
This project predicts house sale prices using the Kaggle House Prices dataset.
## Data Cleaning
- Handled missing values (e.g., LotFrontage imputed by Neighborhood median).
- Removed outliers using IQR on SalePrice.
## Modeling
- Used Gradient Boosting Regressor with max_depth=100, n_estimators=75, learning_rate=0.05.
- Features: LotArea, YearBuilt, YearRemodAdd, OverallQual, OverallCond, OverallQual_Remod, OverallQual_Squared, etc.
## Results
- Max Predicted Value: $313,618.05
- Mean Squared Error (original scale): $426,411,977.67
- Sample Predictions: See predictions_houses.csv
## Limitations
The model fits the dataset range ($755,000 max) and struggles to extrapolate to $1M without additional high-value data.
## Future Work
- Collect data with houses > $1M for better extrapolation.
- Explore advanced models (e.g., XGBoost, neural networks).
