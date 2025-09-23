import json
import joblib
import pandas as pd
import numpy as np
from kafka import KafkaConsumer
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Keep lists to store predictions and actuals
predictions = []
actuals = []
# ---------------- CONFIG ----------------
TOPIC = "air_quality"
BOOTSTRAP = "localhost:9092"

# Rolling/lag features to ffill/bfill
ROLLING_COLS = [
    "CO(GT)_lag_1","CO(GT)_lag_3","CO(GT)_lag_6",
    "CO(GT)_lag_12","CO(GT)_lag_24","CO_roll_std","CO_roll_max",
    "CO(GT)_rollmean_3","CO(GT)_rollstd_3","CO(GT)_rollmean_24","CO(GT)_rollstd_24"
]


# ---------------- LOAD MODEL RANDOM FOREST----------------

#Random Forest
# print("Loading RF model, feature list, and scaler...")
# model = joblib.load("rf_model.joblib")
# scaler = joblib.load("rf_scaler.joblib")

# with open("rf_feature_list.json", "r") as f:
#     feature_cols = json.load(f)

# ---------------- LOAD MODEL XG BOOST----------------

#XG Boost
print("Loading XGB model, feature list, and scaler...")
model = joblib.load("xgb_model.joblib")

scaler = joblib.load("xgb_scaler.joblib")

with open("xgb_feature_list.json", "r") as f:
    feature_cols = json.load(f)

# ---------------- PREPROCESS MESSAGE ----------------
def prepare_message(message, feature_cols):
    """Convert JSON message to DataFrame row with required features."""
    df = pd.DataFrame([message])

    # Drop extra columns not needed by model
    extra_cols = ["Date","Time","CO(GT)","datetime"]
    df = df.drop(columns=[c for c in extra_cols if c in df.columns], errors='ignore')

    # Add missing columns from feature_cols
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan  # initialize with NaN

    # Reorder columns to match training order
    df = df[feature_cols]

    # Apply scaler
    df_scaled = scaler.transform(df)
    return df_scaled

def update_performance(y_true, y_pred):
    """Calculate and print performance metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"📊 Current Holdout Performance: MAE={mae:.4f}, RMSE={rmse:.4f}")


# ---------------- MAIN LOOP ----------------
def main():
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        group_id="air_quality_group",
        auto_offset_reset="earliest",
        enable_auto_commit=True
    )

    print(f"Consuming from topic '{TOPIC}' and making predictions...")
    for message in consumer:
        record = message.value
        print(f"📩 Received: {record}")

        try:
            # Prepare message for model
            X = prepare_message(record, feature_cols)
            pred = model.predict(X)[0]

            print(f"✅ Predicted CO = {pred:.4f}")
        except Exception as e:
            print(f"⚠️ Error processing record: {e}")
    
    try:
        X = prepare_message(record, feature_cols)
        pred = model.predict(X)[0]
        print(f"✅ Predicted CO = {pred:.4f}")

    # If actual CO(GT) is in message, update performance
        if "CO(GT)" in record:
            actual = record["CO(GT)"]
            predictions.append(pred)
            actuals.append(actual)
            update_performance(actuals, predictions)

    except Exception as e:
        print(f"⚠️ Error processing record: {e}")

if __name__ == "__main__":
    main()






