import json
import time
import pandas as pd
from kafka import KafkaProducer

TOPIC = "air_quality"
BOOTSTRAP = "localhost:9092"

def main():
    # Create Kafka producer
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")  # serialize dict to JSON
    )

    # Load holdout dataset for Random Forest and XG Boost
    #df = pd.read_csv("holdout_data.csv")
    
    # Load holdout dataset for Sarima
    df = pd.read_csv("sarima_holdout_data.csv")

    print(f"Streaming {len(df)} messages to topic '{TOPIC}'...")
    for _, row in df.iterrows():
        message = row.to_dict()
        print("Sending:", message)  # preview

        try:
            producer.send(TOPIC, value=message) 
        except Exception as e:
            print(f"Send failed: {e}")
        time.sleep(0.2)  # Simulate ~5 messages/sec

    producer.flush()
    print("Done streaming.")

if __name__ == "__main__":
    main()

