import pandas as pd
import joblib
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load model, scaler, and feature list
model = joblib.load("xgb_model.joblib")
scaler = joblib.load("xgb_scaler.joblib")
with open("xgb_feature_list.json", "r") as f:
    feature_cols = json.load(f)

# Load holdout data
df = pd.read_csv("sarima_holdout_data.csv")

# Keep ground truth
y_true = df["CO(GT)"]

# Select only feature columns that model expects
X = df[feature_cols]

# Apply the same scaler as training
X_scaled = scaler.transform(X)

# Predict
y_pred = model.predict(X_scaled)

# Evaluate
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"Holdout Performance: MAE={mae:.4f}, RMSE={rmse:.4f}")
