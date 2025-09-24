# Troubleshooting Guidelines

This section provides guidance for resolving common issues while running the Real-Time Air Quality Prediction pipeline using Apache Kafka, Docker, and machine learning models.

## 1. Kafka Connectivity Issues

Ensure Docker containers for Kafka and Zookeeper are running:

docker ps


Check Kafka broker logs for errors:

docker logs <kafka_container_name>


Confirm network settings and port mappings (default: 9092) are correctly configured.

## 2. Topic and Partition Problems

Verify topics exist:

docker exec -it <kafka_container_name> kafka-topics --list --bootstrap-server localhost:9092


Ensure the correct topic names are used in producer.py and consumer.py.

If messages aren’t being consumed, check partition assignment and offsets.

## 3. Data Path and File Issues

Confirm CSV paths in producer scripts are correct.

Use the appropriate holdout dataset:

holdout_data.csv for Random Forest/XGBoost

sarima_holdout_data.csv for SARIMA

Check that model artifacts (.joblib, .json) are present and match the training version.

## 4. Feature Mismatch

Ensure incoming Kafka message features match the training feature order:

df = df[feature_cols]


Missing or misordered features may cause inaccurate predictions or runtime errors.

## 5. Library/Dependency Conflicts

Use a consistent Python environment (conda or venv).

Ensure compatible versions of kafka-python, pandas, scikit-learn, xgboost, and statsmodels.

## 6. Streaming Performance / Latency

Streaming large datasets row-by-row may cause lag.

Consider batching or optimizing preprocessing in the consumer script to improve throughput.

## 7. SARIMA Convergence Warnings

Warnings like:

Maximum Likelihood optimization failed to converge are common.

Mitigate by adjusting the SARIMA model order, scaling exogenous variables, or increasing optimization iterations.

## 8. Evaluation Metrics Consistency

Ensure MAE and RMSE calculations in Prediction_performance_eval.py respect any scaling applied during training.

Collect all predictions before calculating aggregate metrics to avoid partial results.

## 9. Docker-Compose Specific Issues

Restart services if containers hang:

docker-compose down && docker-compose up -d


Clean dangling images and volumes if Kafka fails to start:

docker system prune -f

## 10. Logging and Debugging

Add debug prints in consumer.py to confirm the features being read and passed to the model.

Log messages and prediction outputs to a file for post-analysis:

with open("predictions_log.csv", "a") as f:
    f.write(f"{y_true},{y_pred}\n")
