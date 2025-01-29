import random
from datetime import datetime, timezone
import time
import json
from kafka import KafkaProducer, KafkaConsumer
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from prometheus_client import start_http_server, Counter, Gauge
from threading import Thread, Lock
import pandas as pd
from fastapi import FastAPI
from uvicorn import run
import joblib

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

# API endpoints and other constants
api_endpoints = [
    "GET /api/users HTTP/1.1",
    "POST /api/login HTTP/1.1",
    "PUT /api/update HTTP/1.1",
    "DELETE /api/resource HTTP/1.1",
    "GET /api/status HTTP/1.1"
]

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

# Prometheus metrics
log_counters = {
    "INFO": Counter("log_info_total", "Count of INFO logs"),
    "DEBUG": Counter("log_debug_total", "Count of DEBUG logs"),
    "WARN": Counter("log_warn_total", "Count of WARN logs"),
    "ERROR": Counter("log_error_total", "Count of ERROR logs")
}

ANOMALY_COUNT = Gauge("anomaly_count", "Count of detected anomalies")

# Thread lock for shared resources
lock = Lock()

# Kafka setup
KAFKA_TOPIC = 'logs_topic'
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Elasticsearch setup
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

class LogProcessor:
    def __init__(self):
        self.model = self.load_ml_model()
        if self.model is None:
            raise ValueError("Failed to initialize ML model")
    
    @staticmethod
    def load_ml_model():
        try:
            model = joblib.load('log_prediction_model.pkl')
            if not hasattr(model, 'predict'):
                raise AttributeError("Loaded model does not have 'predict' method")
            if not hasattr(model, 'feature_names_in_'):
                raise AttributeError("Model appears to be missing required attributes")
            return model
        except Exception as e:
            print(f"Error loading ML model: {e}")
            return None
    
    def process_log(self, log_data):
        try:
            df = pd.DataFrame([log_data])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            features = {
                'time_since_last_error': 0,
                'rolling_error_count': 1 if log_data['log_level'] == 'ERROR' else 0,
                'rolling_warning_count': 1 if log_data['log_level'] == 'WARN' else 0,
                'error_after_warning': 0,
                'motif_count': 0
            }
            
            input_features = pd.DataFrame([features])
            input_features = input_features[self.model.feature_names_in_]
            
            prediction = self.model.predict_proba(input_features)
            return prediction[0][1]
        except Exception as e:
            print(f"Error during anomaly detection: {e}")
            return 0

class AnomalyDetector:
    def __init__(self, model_path):
        try:
            self.model = joblib.load(model_path)
            if not hasattr(self.model, 'predict'):
                raise AttributeError("Loaded model does not have 'predict' method")
        except Exception as e:
            print(f"Error loading ML model: {e}")
            raise

    def detect_anomaly(self, log_data):
        try:
            features = {
                'rolling_warning_count': 1 if log_data['log_level'] == 'WARN' else 0,
                'error_after_warning': 0,
                'motif_count': 0
            }
            
            input_features = pd.DataFrame([features])
            input_features = input_features[self.model.feature_names_in_]
            
            prediction = self.model.predict_proba(input_features)
            return prediction[0][1]
        except Exception as e:
            print(f"Error during anomaly detection: {e}")
            return 0
        
def generate_log_entry():
    ip = random.choice(ip_addresses)
    timestamp = datetime.now(timezone.utc).isoformat()
    user_agent = random.choice(user_agents)
    api_endpoint = random.choice(api_endpoints)
    
    log_level = random.choices(
        ["INFO", "DEBUG", "WARN", "ERROR"],
        weights=[40, 40, 15, 5],
        k=1
    )[0]
    
    response_code = response_codes.get(log_level, 500)
    
    if log_level in ["INFO", "DEBUG"]:
        message = random.choice(info_debug_messages).format(random.randint(1, 100))
    elif log_level == "WARN":
        message = random.choice(benign_warnings + critical_warnings).format(
            random.choice(["A", "B", "C", "D", "E"])
        )
    else:
        message = random.choice(critical_warnings)
    
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
    
    producer.send(KAFKA_TOPIC, value=log_data)
    return log_data

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

def consume_logs_and_send_to_elasticsearch():
    try:
        log_processor = LogProcessor()
        
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers='localhost:9092',
            group_id='log_consumer_group',
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='earliest',
            enable_auto_commit=True
        )
        
        logs_batch = []
        batch_size = 100
        index_name = "logs"
        
        create_index_with_mapping(index_name)
        
        for message in consumer:
            log_data = message.value
            logs_batch.append(log_data)
            
            anomaly_score = log_processor.process_log(log_data)
            if anomaly_score > 0.6:
                ANOMALY_COUNT.inc()
                print(f"Anomaly detected: {log_data}")
            
            if len(logs_batch) >= batch_size:
                bulk_send_to_elasticsearch(logs_batch)
                logs_batch.clear()
        
        if logs_batch:
            bulk_send_to_elasticsearch(logs_batch)
            
    except Exception as e:
        print(f"Error in consumer: {e}")

def start_prometheus_server():
    start_http_server(8000)
    print("Prometheus server is running on http://localhost:8000")

def start_log_generation():
    flush_interval = 5
    last_flush_time = time.time()
    
    while True:
        log_data = generate_log_entry()
        print(log_data)
        
        if time.time() - last_flush_time >= flush_interval:
            producer.flush()
            last_flush_time = time.time()
        
        time.sleep(1)

app = FastAPI()

@app.post("/predict/")
def predict(log: dict):
    """Endpoint to predict anomalies based on log data"""
    log_processor = LogProcessor()
    anomaly_score = log_processor.process_log(log)
    return {"anomaly_score": anomaly_score}

if __name__ == "__main__":
    start_prometheus_server()
    
    consumer_thread = Thread(target=consume_logs_and_send_to_elasticsearch)
    consumer_thread.start()
    
    api_thread = Thread(target=lambda: run(app, host="0.0.0.0", port=8001, log_level="info"))
    api_thread.start()
    
    start_log_generation()
