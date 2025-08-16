import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from joblib import dump

# Load the training data
train_df = pd.read_csv('/content/train.csv', on_bad_lines='skip')
print("Data loaded successfully!")
print(f"Shape of the dataset: {train_df.shape}")
print (train_df)

# Display first few rows
print("\nFirst 5 rows of the dataset:")
print(train_df.head())
# Basic info
print("\nDataset info:")
print(train_df.info())
# Descriptive statistics
print("\nDescriptive statistics:")
print(train_df.describe(include='all'))

# Drop ID column if exists
if 'id' in train_df.columns:
    train_df.drop('id', axis=1, inplace=True)

# Handle missing values
print("\nMissing values before handling:")
print(train_df.isnull().sum())

# For simplicity, we'll fill numerical missing values with median and categorical with mode
for col in train_df.columns:
    if train_df[col].dtype in ['int64', 'float64']:
        train_df[col].fillna(train_df[col].median(), inplace=True)
    else:
        train_df[col].fillna(train_df[col].mode()[0], inplace=True)

print("\nMissing values after handling:")
print(train_df.isnull().sum())

# Convert date column to datetime and extract features (if date column exists)
date_col = None
for col in train_df.columns:
    if 'date' in col.lower():
        date_col = col
        break

if date_col:
    train_df[date_col] = pd.to_datetime(train_df[date_col], errors='coerce', format='mixed')
    train_df['year'] = train_df[date_col].dt.year
    train_df['month'] = train_df[date_col].dt.month
    train_df['day'] = train_df[date_col].dt.day
    train_df.drop(date_col, axis=1, inplace=True)
    print(f"\nProcessed date column: {date_col}")

# Label Encoding for categorical columns
label_encoders = {}
for col in train_df.columns:
    if train_df[col].dtype == 'object':
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        label_encoders[col] = le
        print(f"Label encoded column: {col}")
print("\nData after preprocessing:")
print(train_df.head())

import gc
gc.collect()

# Separate features and target
X = train_df.drop('Premium Amount', axis=1)
y = train_df['Premium Amount']

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_val)
print (y_pred)

# Calculate metrics
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred)
print("\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.show()

# Train final model on all data
final_model = RandomForestRegressor(n_estimators=100, random_state=42)
final_model.fit(X, y)

mse = mean_squared_error(y, final_model.predict(X))
rmse = np.sqrt(mse)
r2 = r2_score(y, final_model.predict(X))
print("\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")

dump(final_model, 'model.joblib')

dump(label_encoders, 'label_encoders.joblib')