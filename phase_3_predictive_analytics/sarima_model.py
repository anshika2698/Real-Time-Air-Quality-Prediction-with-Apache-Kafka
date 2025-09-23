"""
Offline training + validation for SARIMA forecasting (CO)
Saves: sarima_model.pkl, scaler.joblib, feature_list.json, metrics.json
"""

import json
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# -------------- CONFIG --------------
CSV_PATH = "sarima_training_data.csv"
DATETIME_COL = "datetime"
TARGET = "CO(GT)"
TEST_HOURS = 24 * 14      # last 14 days as test
MODEL_OUT = "sarima_model.joblib"
SCALER_OUT = "sarima_scaler.joblib"
FEATURES_OUT = "sarima_feature_list.json"
METRICS_OUT = "metrics.json"
ORDER = (1, 1, 1)           # (p,d,q) - tune for your case!
SEASONAL_ORDER = (1, 1, 1, 24) # (P,D,Q,s) - s=24 for hourly
# ------------------------------------

EXOG_COLS = ['PT08.S1(CO)', 'C6H6(GT)', 'NOx(GT)', 'PT08.S3(NOx)',
             'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH',
             'AH', 'hour', 'dayofweek', 'month', 'hour_sin', 'hour_cos']

def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
    df = df.sort_values(DATETIME_COL).reset_index(drop=True)
    df = df.set_index(DATETIME_COL)
    df = df.resample("h").mean(numeric_only=True)
    df = df.reset_index()
    return df

def temporal_split(df, test_hours=TEST_HOURS):
    n = len(df)
    split_idx = n - test_hours if test_hours < n else int(n * 0.8)
    train = df.iloc[:split_idx].reset_index(drop=True)
    test = df.iloc[split_idx:].reset_index(drop=True)
    return train, test

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

def main():
    print("Loading and preparing data...")
    df = load_and_prepare(CSV_PATH)

    # Handling NaNs in target for SARIMA
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    # Handling NaNs in exogenous features by median
    for c in EXOG_COLS:
        df[c] = df[c].fillna(df[c].median())

    # Train/test split
    train, test = temporal_split(df)
    X_train_exog = train[EXOG_COLS]
    X_test_exog = test[EXOG_COLS]
    y_train = train[TARGET]
    y_test = test[TARGET]

    # Feature scaling for exogenous variables (fit on train, transform on test)
    scaler = StandardScaler()
    X_train_exog_scaled = scaler.fit_transform(X_train_exog)
    X_test_exog_scaled = scaler.transform(X_test_exog)

    # Adding training medians
    training_medians = pd.Series({col: scaler.mean_[i] for i, col in enumerate(EXOG_COLS)})

    # Save medians for consumer
    training_medians.to_json("sarima_training_medians.json")

    # Baseline: naive lag-1
    y_base = test[TARGET].shift(1)
    y_test_valid = y_test[1:]
    y_base_valid = y_base[1:]
    base_mae, base_rmse = (None, None)
    if len(y_test_valid) > 0:
        base_mae, base_rmse = evaluate(y_test_valid, y_base_valid)
        print(f"Baseline (lag-1) MAE={base_mae:.4f}, RMSE={base_rmse:.4f}")

    # SARIMAX fit (with exogenous)
    print("Training SARIMAX model...")
    model = SARIMAX(y_train, order=ORDER, seasonal_order=SEASONAL_ORDER,
                    exog=X_train_exog_scaled,
                    enforce_stationarity=False, enforce_invertibility=False)
    sarima_res = model.fit(disp=False)

    # SARIMAX forecast (provide exog for test period)
    forecast = sarima_res.forecast(steps=len(X_test_exog_scaled), exog=X_test_exog_scaled)
    mae, rmse = evaluate(y_test, forecast)
    print(f"SARIMA Test MAE={mae:.4f}, RMSE={rmse:.4f}")

    # Save model, scaler, columns, and metrics
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(sarima_res, f)
    joblib.dump(scaler, SCALER_OUT)
    with open(FEATURES_OUT, "w") as f:
        json.dump(EXOG_COLS, f)
    metrics = {
        "baseline_mae": base_mae,
        "baseline_rmse": base_rmse,
        "sarima_mae": mae,
        "sarima_rmse": rmse,
        "order": ORDER,
        "seasonal_order": SEASONAL_ORDER,
        "exog_cols": EXOG_COLS
    }
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved SARIMA model and metadata.")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
