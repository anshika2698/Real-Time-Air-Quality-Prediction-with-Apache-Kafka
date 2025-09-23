import json
import joblib
import pickle
import pandas as pd
import numpy as np
from kafka import KafkaConsumer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import traceback

# ---------------- CONFIG ----------------
TOPIC = "air_quality"
BOOTSTRAP = "localhost:9092"
TARGET = "CO(GT)"

# Exogenous features used during SARIMA training
EXOG_COLS = [
    "PT08.S1(CO)", "C6H6(GT)", "NOx(GT)", "PT08.S3(NOx)",
    "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH",
    "AH", "hour", "dayofweek", "month", "hour_sin", "hour_cos"
]

MODEL_PATH = "sarima_model.joblib"
SCALER_PATH = "sarima_scaler.joblib"

# ---------------- LOAD MODEL AND SCALER ----------------
print("Loading SARIMA model and scaler...")
sarima_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Use training scaler means for NaN imputation
training_medians = {col: scaler.mean_[i] for i, col in enumerate(EXOG_COLS)}

# ---------------- PERFORMANCE TRACKERS ----------------
predictions = []
actuals = []

def update_metrics():
    if len(predictions) > 0:
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        print(f"Current performance - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# ---------------- EXOG FEATURE PREPARATION ----------------
def prepare_exog(message):
    """Prepare exogenous variables from Kafka message with proper DataFrame and NaN handling."""
    feature_dict = {}
    for col in EXOG_COLS:
        val = message.get(col, np.nan)
        if val is None or pd.isna(val):
            val = training_medians[col]  # impute missing
        feature_dict[col] = val

    # Convert to DataFrame with proper column names
    df_exog = pd.DataFrame([feature_dict], columns=EXOG_COLS)

    # Scale features
    exog_scaled = scaler.transform(df_exog)
    return exog_scaled[0]

# ---------------- MAIN CONSUMER LOOP ----------------
def main():
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        group_id="sarima_forecasting_group",
        auto_offset_reset="earliest"
    )

    print(f"Consuming topic '{TOPIC}' for SARIMA forecasts...")

    # Initial target history from training
    history_y = list(sarima_model.model.endog)
    history_exog = []

    for message in consumer:
        record = message.value
        print(f"Received: {record}")

        try:
            exog_next = prepare_exog(record)
            history_exog.append(exog_next)

            # Forecast next point using SARIMA with exogenous features
            forecast = sarima_model.get_forecast(steps=1, exog=[exog_next])
            pred = forecast.predicted_mean.iloc[0]

            print(f"Predicted CO(GT): {pred:.4f}")

            predictions.append(pred)

            # Update metrics if actual target available
            if TARGET in record and record[TARGET] is not None:
                actual_val = record[TARGET]
                actuals.append(actual_val)
                history_y.append(actual_val)

                # Keep history length same as training to prevent memory issues
                max_history_len = len(sarima_model.model.endog)
                if len(history_y) > max_history_len:
                    history_y.pop(0)
                if len(history_exog) > max_history_len:
                    history_exog.pop(0)

                update_metrics()
            else:
                print("No actual CO(GT) in message, skipping metrics update.")

        except Exception as e:
            print(f"Error processing record: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()

