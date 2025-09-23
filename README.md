# Real-Time-Air-Quality-Prediction-with-Apache-Kafka
This project uses Apache Kafka for real-time data streaming and environmental time series analysis. Develop an end-to-end data pipeline using the UCI Air Quality dataset to demonstrate merit in streaming architecture, exploratory data analysis, and predictive modeling deployment. 

# Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [Contributing](#contributing)

# Dataset
- The dataset contains hourly air quality measurements from a gas multisensor device deployed on the field in an Italian city from March 2004 to February 2005.
- Features include: Date	(DD/MM/YYYY), Time	(HH.MM.SS), True hourly averaged concentration CO in mg/m^3  (reference analyzer), PT08.S1 (tin oxide)  hourly averaged sensor response (nominally  CO targeted), True hourly averaged overall Non Metanic HydroCarbons concentration in microg/m^3 (reference analyzer), True hourly averaged Benzene concentration  in microg/m^3 (reference analyzer), PT08.S2 (titania) hourly averaged sensor response (nominally NMHC targeted), True hourly averaged NOx concentration  in ppb (reference analyzer), PT08.S3 (tungsten oxide) hourly averaged sensor response (nominally NOx targeted), True hourly averaged NO2 concentration in microg/m^3 (reference analyzer), PT08.S4 (tungsten oxide) hourly averaged sensor response (nominally NO2 targeted), PT08.S5 (indium oxide) hourly averaged sensor response (nominally O3 targeted), Temperature in Â°C, Relative Humidity (%), AH Absolute Humidity
- Source: [UCI Machine Learning Repository - Air Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Air+Quality)

# Installation

Docker Installation

-Visit [Docker Desktop](https://www.docker.com/products/docker-desktop/) and download the version for your OS (Windows/macOS/Linux).
- Follow the installation instructions on the website.
  
Verify Installation
 ```bash 
 docker --version
 docker-compose --version
  ```
- Start Docker by launching Docker Desktop

Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/repo.git](https://github.com/anshika2698/Real-Time-Air-Quality-Prediction-with-Apache-Kafka.git)
   cd repo
   ```
Note: the docker-compose.yml file in phase_1_streaming_infrastructure doc will help you setup kafka in 

- Run Docker Compose to Start Kafka
  ```bash
  docker-compose up -d
  ```
- Check running containers
   ```bash
  docker ps
  ```
You will see something like:
```bash
CONTAINER ID   IMAGE                            PORTS
xxxxxx         confluentinc/cp-kafka:latest     0.0.0.0:9092->9092/tcp
```
- Once the container is running, create the topic
   ```bash
  docker exec -it <kafka_container_name> \
  kafka-topics --create \
  --topic air_quality \
  --bootstrap-server localhost:9092 \
  --partitions 1 \
  --replication-factor 1
  ```
- List Topics to Confirm
  ```bash
  docker exec -it <kafka_container_name> \
  kafka-topics --list --bootstrap-server localhost:9092
  ```
- You should see "air_quality"
- Now install kafka-python
  ```bash
  pip install kafka-python
  ```
# Usage
 - Once the repository is cloned and the Docker and Kafka setups are working
 Following the given steps to run the model:
## EDA
- Run the EDA.ipynb file present in phase_2 folder, which would give you the required CSV files to run the XGBoost and Random Forest Model.
- Run the sarima_preproccessin.ipynb file in phase_2 folder, which would the required CSV files to run the Sarima Model.
Note:
These are the CSV files you will get: training_data.csv, holdout_data.csv, sarima_training_data.csv and sarima_holdout_data.csv.
## Model Building
- Now go to the phase_3 folder which contains 3 .py files for random forest, xgboost and sarima model.
- Make sure you have the needed libraries
- Make sure the training data file paths are according to your system (xgboost and random forest have the same training_data.csv for training, and sarima uses sarima_training_data.csv.
- Build the models, run the following commands in your terminal: 
  ```bash
  python random_forest_modeltrain.py
  python xgboost_modeltrain.py
  python sarima_model.py
  ```    
- The 3 models should be successfully built
  
## Kafka Producer File

- The producer.py file in phase_1 folder uses the holdout data. We have 2 sets of holdout datasets i) holdout_data.csv (for xgboost and random forest), ii) sarima_holdout_data.csv (for Sarina model)
- Make sure the file paths for the CSV files are correct in your producer.py code.
- Since we are using the same producer file for all three models, make sure you comment the following code snippet if you are running Sarima Model:
```bash
    Load holdout dataset for Random Forest and XG Boost
    df = pd.read_csv("holdout_data.csv")    
  ```
-  If you are running XgBoost or Random Forest Model, make sure to comment the following code snippet:
  ```bash
     Load holdout dataset for Sarima
     df = pd.read_csv("sarima_holdout_data.csv")  
  ```
## Kafka Consumer File
- We use consumer.py file in phase_1 folder for the XgBoost and Random Forest Model, and we use consumer_sarima.py for Sarima model.
- The Kafka consumer file utilizes .joblib and .json files that will be generated after we have run our models.
- These files are called within the consumer logic as they set up the model that will be responsible for real-time predictions with the streaming data.
- Make sure to comment the following code snippets in the consumer.py file if running Random Forest:
  ```bash
     # ---------------- LOAD MODEL XG BOOST----------------

     #XG Boost
     print("Loading XGB model, feature list, and scaler...")
     model = joblib.load("xgb_model.joblib")

     scaler = joblib.load("xgb_scaler.joblib")

     with open("xgb_feature_list.json", "r") as f:
        feature_cols = json.load(f) 
  ```
- Make sure to comment the following code snippets in the consumer.py file if running XG Boosting:
   ```bash
    # ---------------- LOAD MODEL RANDOM FOREST----------------

    #Random Forest
     print("Loading RF model, feature list, and scaler...")
     model = joblib.load("rf_model.joblib")
     scaler = joblib.load("rf_scaler.joblib")

     with open("rf_feature_list.json", "r") as f:
         feature_cols = json.load(f)
 
  ```

## Running The Project

- Once everything above is ready, we can run our producer and consumer files to make real-time predictions for CO
- Split the terminal to see the message being sent by the producer and received by the consumer, and the real-time predictions being performed by the model of your choosing
- Run the following command in Terminal 1:
    ```bash
     python producer.py  
    ```
- Run the following command in Terminal 2: 
    ```bash
     python consumer.py
    ```
- Run the Prediction_performance_eval.py to evaluate the prediction performance of models on the holdout data:
  ```bash
     python Prediction_performance_eval.py
    ```
- Note: using MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error) as the metrics to measuring our model performance

# Results

- Model Performance with Training Data:

```bash
#Random Forest
(base) Anshikas-MacBook-Air:phase_3_predictive_analytics anshikashukla$ python random_forest_modeltrain.py
Loading and preparing data...
Handling NaN values...
Baseline (lag-1) MAE=0.5119, RMSE=0.7670
Training Random Forest with TimeSeriesSplit grid search...
Fitting 3 folds for each of 12 candidates, totalling 36 fits
Best params: {'max_depth': None, 'min_samples_split': 5, 'n_estimators': 200}
RF Test MAE=0.1975, RMSE=0.3507
Saved model and metadata.
{
  "baseline_mae": 0.5119047619047619,
  "baseline_rmse": 0.7669900974092231,
  "rf_mae": 0.19753871008703588,
  "rf_rmse": 0.35066851134184746,
  "best_params": {
    "max_depth": null,
    "min_samples_split": 5,
    "n_estimators": 200
  }
}

#XGBoost
(base) Anshikas-MacBook-Air:phase_3_predictive_analytics anshikashukla$ python xgboost_modeltrain.py
Loading and preparing data...
Handling NaN values...
Baseline (lag-1) MAE=0.5119, RMSE=0.7670
Training XGBoost with TimeSeriesSplit grid search...
Fitting 3 folds for each of 24 candidates, totalling 72 fits
Best params: {'learning_rate': 0.05, 'max_depth': 10, 'n_estimators': 200, 'subsample': 0.8}
XGB Test MAE=0.1794, RMSE=0.3319
Saved XGBoost model and metadata.
{
  "baseline_mae": 0.5119047619047619,
  "baseline_rmse": 0.7669900974092231,
  "xgb_mae": 0.1793936475934017,
  "xgb_rmse": 0.3319392764195327,
  "best_params": {
    "learning_rate": 0.05,
    "max_depth": 10,
    "n_estimators": 200,
    "subsample": 0.8
  }
}

#SARIMA
(base) Anshikas-MacBook-Air:phase_3_predictive_analytics anshikashukla$ python sarima_model.py
Loading and preparing data...
Baseline (lag-1) MAE=0.5063, RMSE=0.7597
Training SARIMAX model...
/opt/anaconda3/lib/python3.12/site-packages/statsmodels/base/model.py:607: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
  warnings.warn("Maximum Likelihood optimization failed to "
SARIMA Test MAE=0.5042, RMSE=0.6593
Saved SARIMA model and metadata.
{
  "baseline_mae": 0.5062686567164179,
  "baseline_rmse": 0.7596935753040229,
  "sarima_mae": 0.5042401337727255,
  "sarima_rmse": 0.6593370591425833,
  "order": [
    1,
    1,
    1
  ],
  "seasonal_order": [
    1,
    1,
    1,
    24
  ],
  "exog_cols": [
    "PT08.S1(CO)",
    "C6H6(GT)",
    "NOx(GT)",
    "PT08.S3(NOx)",
    "NO2(GT)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
    "hour",
    "dayofweek",
    "month",
    "hour_sin",
    "hour_cos"
  ]

```
- Model Performance with Holdout Data:
```bash
      #Random Forest
      Holdout Performance: MAE=1.6428, RMSE=1.9869

      #XGBoost
      Holdout Performance: MAE=0.1512, RMSE=0.2481

      #SARIMA
      Holdout Performance: MAE: 1.0777, RMSE= 1.1542
```
- Note: We see the SARIMA evaluation with each prediction it makes while running the consumer file
- You will see the following output in synchronization in your terminal split screen (producer: left screen; consumer: right screen)

<img width="3116" height="1390" alt="image" src="https://github.com/user-attachments/assets/ea35ac65-bbe8-4862-8996-8120e4277ff9" />

#Contribution
@anshika2698


