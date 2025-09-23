"""
Offline training + validation for Random Forest forecasting (CO)
Saves: rf_model.joblib, scaler.joblib, feature_list.json, metrics.json
"""

import json
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from datetime import timedelta

# -------------- CONFIG --------------
CSV_PATH = "training_data.csv"   # cleaned CSV with 'datetime' column
DATETIME_COL = "datetime"
TARGET = "CO(GT)"
TEST_HOURS = 24 * 14    # last 14 days as test
LAGS = [1, 3, 6, 12, 24]
ROLL_WINDOWS = [3, 24]
MODEL_OUT = "rf_model.joblib"
SCALER_OUT = "rf_scaler.joblib"
FEATURES_OUT = "rf_feature_list.json"
METRICS_OUT = "rf_metrics.json"
RANDOM_STATE = 42
# ------------------------------------

def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)
    # ensure datetime
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
    df = df.sort_values(DATETIME_COL).reset_index(drop=True)

    # set datetime index for resampling
    df = df.set_index(DATETIME_COL)

    # resample to hourly, keep only numeric columns
    df = df.resample("h").mean(numeric_only=True)

    # bring datetime back
    df = df.reset_index()
    return df

def temporal_split(df, test_hours=TEST_HOURS):
    n = len(df)
    if test_hours < n:
        split_idx = n - test_hours
    else:
        split_idx = int(n * 0.8)
    train = df.iloc[:split_idx].reset_index(drop=True)
    test = df.iloc[split_idx:].reset_index(drop=True)
    return train, test

def evaluate(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, rmse

def main():
    print("Loading and preparing data...")
    df = load_and_prepare(CSV_PATH)

    # Handle NaNs
    # -------------------------------
    print("Handling NaN values...")

    #Dropping rows where target itself is missing (cannot train without target)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    #Imputing sensor/environmental features with median
    sensor_cols = ['PT08.S1(CO)', 'C6H6(GT)',
                    'NOx(GT)', 'PT08.S3(NOx)',
                   'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)',
                   'T', 'RH', 'AH']
    for col in sensor_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    #Imputing lag/rolling features with forward-fill then backward-fill
    lag_roll_cols = ['CO(GT)_lag_1','CO(GT)_lag_3','CO(GT)_lag_6','CO(GT)_lag_12',
                     'CO(GT)_lag_24','CO_roll_std','CO_roll_max','CO(GT)_rollmean_3','CO(GT)_rollmean_24']
    df[lag_roll_cols] = df[lag_roll_cols].ffill().bfill()

    #Checking if any NaNs remain
    if df.isna().sum().sum() > 0:
        print("Warning: There are still NaNs remaining in the dataset")
        print(df.isna().sum()[df.isna().sum() > 0])

    # Split train/test
    # -------------------------------
    train, test = temporal_split(df)
    exclude_cols = [DATETIME_COL, TARGET]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X_train = train[feature_cols]
    y_train = train[TARGET]
    X_test = test[feature_cols]
    y_test = test[TARGET]

    # Baseline (naive lag-1)
    # -------------------------------
    y_base = test['CO(GT)_lag_1']
    base_mae, base_rmse = evaluate(y_test, y_base)
    print(f"Baseline (lag-1) MAE={base_mae:.4f}, RMSE={base_rmse:.4f}")

    # Scale features (optional for RF)
    # -------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest training with TimeSeriesSplit CV
    # -------------------------------
    print("Training Random Forest with TimeSeriesSplit grid search...")
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    param_grid = {"n_estimators":[100,200], "max_depth":[8,12,None], "min_samples_split":[2,5]}
    tscv = TimeSeriesSplit(n_splits=3)
    grid = GridSearchCV(rf, param_grid, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    print("Best params:", grid.best_params_)

    # Evaluate model
    # -------------------------------
    y_pred = best.predict(X_test)
    mae, rmse = evaluate(y_test, y_pred)
    print(f"RF Test MAE={mae:.4f}, RMSE={rmse:.4f}")

    # -------------------------------
    # Save model, scaler, features, metrics
    # -------------------------------
    joblib.dump(best, MODEL_OUT)
    joblib.dump(scaler, SCALER_OUT)
    with open(FEATURES_OUT, "w") as f:
        json.dump(feature_cols, f)

    metrics = {
        "baseline_mae": base_mae,
        "baseline_rmse": base_rmse,
        "rf_mae": mae,
        "rf_rmse": rmse,
        "best_params": grid.best_params_
    }
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model and metadata.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()