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
from fastapi import FastAPI
from uvicorn import run
import joblib
from scipy.sparse import hstack
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

# Log messages
info_debug_messages = [
    "User login successful with username: user{}. ",
    "Cache cleared successfully for session ID {}.",
    "Health check passed for service '{}'.",
    "Fetched user data successfully.",
    "Connected to database successfully."
]

benign_warnings = [
    "User entered incorrect password multiple times.",
    "API rate limit approaching threshold for IP: {}.",
    "Failed login attempt detected for username: user{}. ",
    "Temporary network glitch detected during API call.",
    "Disk usage exceeded 75%, monitoring closely."
]

critical_warnings = [
    "Server network latency detected during operation.",
    "Server CPU temperature is about to reach threshold.",
    "Memory usage is critically high.",
    "Disk space is critically low.",
    "Database connection timeout."
]

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
    with open('model_and_features.pkl', 'rb') as f:
        model_and_features = pickle.load(f)
        model = model_and_features.get('model')
        print("ML model loaded successfully")
except Exception as e:
    print(f"Error loading ML model: {e}")
    model = None

# Function to process logs through the ML model for anomaly detection
# HEM EDITED: Added ML model integration to process Kafka logs for anomalies
def process_log_for_anomaly(log_data):
    try:
        # Convert the log data to a DataFrame
        df = pd.DataFrame([log_data])  # Wrap in a list to create a single-row DataFrame
       
        # Ensure the model receives only the required features
        required_features = ['timestamp', 'ip', 'api_endpoint', 'log_level', 'user_agent', 'response_code']
        input_features = df[required_features]
       
        # Vectorize the log message using the motif_vectorizer and tfidf_vectorizer
        # Transform the log message into features similar to the training data
        message = log_data.get('message', '')
        motif_features = model_and_features['motif_vectorizer'].transform([message])
        tfidf_features = model_and_features['tfidf_vectorizer'].transform([message])

        # Combine both features
        X_input = hstack([tfidf_features, motif_features])

        # Predict with the model
        prediction = model.predict(X_input)
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

    # Adjusted weights to include ERROR logs (5%)
    log_level = random.choices(
        ["INFO", "DEBUG", "WARN", "ERROR"],
        weights=[40, 40, 15, 5],
        k=1
    )[0]

    response_code = response_codes.get(log_level, 500)  # Default to 500 for ERROR

    if log_level in ["INFO", "DEBUG"]:
        message = random.choice(info_debug_messages).format(random.randint(1, 100))
    elif log_level == "WARN":
        message = random.choice(benign_warnings + critical_warnings).format(
            random.choice(["A", "B", "C", "D", "E"])
        )
    elif log_level == "ERROR":
        message = random.choice(critical_warnings)

    # Increment Prometheus counter safely
    with lock:
        log_counters[log_level].inc()

    log_data = {
        "timestamp": timestamp,
        "ip": ip,
        "api_endpoint": api_endpoint,
        "log_level": log_level,
        "user_agent": user_agent,
        "response_code": response_code,
        "message": message
    }

    # Send log to Kafka
    producer.send(KAFKA_TOPIC, value=log_data)

    return log_data


# Function to create an index with mapping in Elasticsearch
def create_index_with_mapping(index_name):
    if not es.indices.exists(index=index_name):
        mapping = {
            "mappings": {
                "properties": {
                    "timestamp": {"type": "date"},
                    "ip": {"type": "ip"},
                    "api_endpoint": {"type": "keyword"},
                    "log_level": {"type": "keyword"},
                    "user_agent": {"type": "text"},
                    "response_code": {"type": "integer"},
                    "message": {"type": "text"}
                }
            }
        }
        es.indices.create(index=index_name, body=mapping)
        print(f"Index '{index_name}' created with mapping")

# Bulk index logs to Elasticsearch
def bulk_send_to_elasticsearch(logs):
    actions = [
        {
            "_index": "logs",
            "_source": log
        }
        for log in logs
    ]
    success, _ = bulk(es, actions)
    print(f"Bulk index: {success} logs sent to Elasticsearch")

    
# Function to consume logs, process for anomalies, and send to Elasticsearch
# HEM EDITED: Enhanced consumer logic to integrate anomaly detection
# HEM EDITED: Sends processed logs and anomalies to Elasticsearch
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
    index_name = "logs"

    create_index_with_mapping(index_name)  # Ensure index exists

    for message in consumer:
        log_data = message.value
        logs_batch.append(log_data)

        # Process log for anomaly detection
        anomaly_score = process_log_for_anomaly(log_data)
        if anomaly_score > 0.6: # Example threshold for anomaly detection
            ANOMALY_COUNT.inc()
            print(f"Anomaly detected: {log_data}")

        if len(logs_batch) >= batch_size:
            bulk_send_to_elasticsearch(logs_batch)
            logs_batch.clear()

    if logs_batch:
        bulk_send_to_elasticsearch(logs_batch)

# HEM ADDED: Start Prometheus HTTP server for metrics
# HEM ADDED: Expose Prometheus metrics for visualization
def start_prometheus_server():
    start_http_server(8000)
    print("Prometheus server is running on http://localhost:8000")

# Start log generation
# HEM EDITED: Generate logs and flush to Kafka periodically
def start_log_generation():
    flush_interval = 5  # Flush every 5 seconds
    last_flush_time = time.time()

    while True:
        log_data = generate_log_entry()
        print(log_data)

        if time.time() - last_flush_time >= flush_interval:
            producer.flush()
            last_flush_time = time.time()

        time.sleep(1)

# HEM ADDED: FastAPI application to expose anomaly detection API
app = FastAPI()

@app.post("/predict/")
def predict(log: dict):
    """Endpoint to predict anomalies based on log data"""
    anomaly_score = process_log_for_anomaly(log)
    return {"anomaly_score": anomaly_score}

# Main function
# HEM EDITED: Updated to include API, consumer, and log generation
if __name__ == "__main__":
    start_prometheus_server()

    # Start Kafka consumer thread
    consumer_thread = Thread(target=consume_logs_and_send_to_elasticsearch)
    consumer_thread.start()

    # Start FastAPI server thread
    api_thread = Thread(target=lambda: run(app, host="0.0.0.0", port=8001, log_level="info"))
    api_thread.start()

    # Start log generation
    start_log_generation()