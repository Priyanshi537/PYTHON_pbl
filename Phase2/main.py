import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# Step 1: Load dataset with correct separator
data = pd.read_csv("AirQualityUCI.csv", sep=';')
print("Shape:", data.shape)
print("Columns:", data.columns)
print(data.head())
# Step 2: Remove empty columns
data = data.dropna(axis=1, how='all')

# Step 3: Replace commas in numbers and convert to float
data = data.replace(',', '.', regex=True)

# Step 4: Convert all columns (except Date/Time) to numeric
for col in data.columns:
    if col not in ['Date', 'Time']:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Step 5: Drop missing values
data = data.dropna()

# Step 6: Define features and target
# Example: Predict CO(GT)
X = data.drop(['CO(GT)', 'Date', 'Time'], axis=1)
y = data['CO(GT)']

# Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 8: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 9: Predict
y_pred = model.predict(X_test)

# Step 10: Evaluate
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

# Step 11: Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained & saved successfully!")