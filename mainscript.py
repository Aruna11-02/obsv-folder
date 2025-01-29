import random
from datetime import datetime, timezone
import time
import json
from kafka import KafkaProducer, KafkaConsumer
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from prometheus_client import start_http_server, Counter, Gauge
from threading import Thread, Lock
import pickle
import pandas as pd
import six  # Ensure 'six' is imported

# Predefined IP addresses
ip_addresses = [
    "192.168.1.1", "192.168.1.2", "192.168.1.3", "192.168.1.4", "192.168.1.5",
    "192.168.1.6", "192.168.1.7", "192.168.1.8", "192.168.1.9", "192.168.1.10"
]

# User agents for realism
user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Mobile/15E148 Safari/604.1"
]

# API endpoints for HTTP requests
api_endpoints = [
    "GET /api/users HTTP/1.1",
    "POST /api/login HTTP/1.1",
    "PUT /api/update HTTP/1.1",
    "DELETE /api/resource HTTP/1.1",
    "GET /api/status HTTP/1.1"
]

# Response codes mapped to log levels
response_codes = {
    "INFO": 200,
    "DEBUG": 200,
    "WARN": 400,
    "ERROR": 404
}

# Prometheus counters to track the log levels
log_counters = {
    "INFO": Counter("log_info_total", "Count of INFO logs"),
    "DEBUG": Counter("log_debug_total", "Count of DEBUG logs"),
    "WARN": Counter("log_warn_total", "Count of WARN logs"),
    "ERROR": Counter("log_error_total", "Count of ERROR logs")
}

# Prometheus gauge for anomaly counts
ANOMALY_COUNT = Gauge("anomaly_count", "Count of detected anomalies")

# Thread lock for shared resources
lock = Lock()

# Kafka Producer setup
try:
    producer = KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    print("Connected to Kafka")
except Exception as e:
    print(f"Kafka producer error: {e}")
    exit(1)

# Elasticsearch setup
try:
    es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
    if not es.ping():
        raise ValueError("Connection to Elasticsearch failed")
    print("Connected to Elasticsearch")
except Exception as e:
    print(f"Elasticsearch connection error: {e}")
    exit(1)

# Kafka Topic for logs
KAFKA_TOPIC = 'logs_topic'

# Load the ML model
try:
    with open('log_prediction_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except EOFError:
    print("Error loading ML model: Ran out of input - The file might be empty or corrupted")
except FileNotFoundError:
    print("Error loading ML model: File not found")
except Exception as e:
    print(f"Error loading ML model: {e}")
    exit(1)

# Function to process logs through the ML model for anomaly detection
def process_log_for_anomaly(log_data):
    if model is None:
        print("ML model is not loaded. Skipping anomaly detection.")
        return 0  # Default score if model is unavailable

    try:
        df = pd.DataFrame([log_data])
        prediction = model.predict(df)
        return prediction[0]  # Return the first prediction
    except Exception as e:
        print(f"Error during anomaly detection: {e}")
        return 0

# Function to generate a log entry
def generate_log_entry():
    ip = random.choice(ip_addresses)
    timestamp = datetime.now(timezone.utc).isoformat()  # Updated to include timezone
    user_agent = random.choice(user_agents)
    api_endpoint = random.choice(api_endpoints)
    log_level = random.choices(
        ["INFO", "DEBUG", "WARN", "ERROR"],
        weights=[40, 40, 15, 5],
        k=1
    )[0]
    response_code = response_codes.get(log_level, 500)  # Default to 500 for ERROR

    log_counters[log_level].inc()  # Increment Prometheus counter safely

    log_data = {
        "timestamp": timestamp,
        "ip": ip,
        "api_endpoint": api_endpoint,
        "log_level": log_level,
        "user_agent": user_agent,
        "response_code": response_code
    }

    producer.send(KAFKA_TOPIC, value=log_data)  # Send log to Kafka
    return log_data

# Bulk index logs to Elasticsearch
def bulk_send_to_elasticsearch(logs):
    actions = [
        {
            "_index": "logs",
            "_source": log
        }
        for log in logs
    ]
    bulk(es, actions)

# Consume logs from Kafka and send to Elasticsearch
def consume_logs_and_send_to_elasticsearch():
    try:
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers='localhost:9092',
            group_id='log_consumer_group',
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='earliest',
            enable_auto_commit=True
        )
        print("Connected to Kafka consumer")
    except Exception as e:
        print(f"Error initializing Kafka consumer: {e}")
        return

    logs_batch = []
    batch_size = 100

    for message in consumer:
        log_data = message.value
        logs_batch.append(log_data)

        anomaly_score = process_log_for_anomaly(log_data)
        if anomaly_score > 0.8:
            ANOMALY_COUNT.inc()  # Increment anomaly count

        if len(logs_batch) >= batch_size:
            bulk_send_to_elasticsearch(logs_batch)
            logs_batch.clear()

    if logs_batch:
        bulk_send_to_elasticsearch(logs_batch)

# Start Prometheus server for metrics
def start_prometheus_server():
    start_http_server(8000)  # Expose metrics at http://localhost:8000

# Main function to run threads
def main():
    start_prometheus_server()

    consumer_thread = Thread(target=consume_logs_and_send_to_elasticsearch)
    consumer_thread.start()

    while True:
        log_data = generate_log_entry()
        print(f"Generated log: {log_data}")
        time.sleep(1)  # Simulate log generation interval

if __name__ == "__main__":
    main()