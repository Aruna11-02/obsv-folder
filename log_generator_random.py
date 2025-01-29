import random
from datetime import datetime, timezone
import pandas as pd
import json

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
    "User login successful with username: user{}.",
    "Cache cleared successfully for session ID {}.",
    "Health check passed for service '{}'.",
    "Fetched user data successfully.",
    "Connected to database successfully."
]

benign_warnings = [
    "User entered incorrect password multiple times.",
    "API rate limit approaching threshold for IP: {}.",
    "Failed login attempt detected for username: user{}.",
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

# Function to generate a log entry
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
        message = random.choice(benign_warnings + critical_warnings).format(random.choice(["A", "B", "C", "D", "E"]))
    elif log_level == "ERROR":
        message = random.choice(critical_warnings)

    log_data = {
        "timestamp": timestamp,
        "ip": ip,
        "api_endpoint": api_endpoint,
        "log_level": log_level,
        "user_agent": user_agent,
        "response_code": response_code,
        "message": message
    }

    return log_data

# Function to generate logs in bulk
def generate_logs(num_logs):
    logs = [generate_log_entry() for _ in range(num_logs)]
    return logs

# Save logs to a JSON file
def save_logs_to_file(logs, filename="logs.json"):
    with open(filename, "w") as file:
        json.dump(logs, file, indent=4)

# Save logs to a DataFrame
def save_logs_to_dataframe(logs):
    df = pd.DataFrame(logs)
    df.to_csv("logs.csv", index=False)
    return df

# Main
if __name__ == "__main__":
    num_logs = 1000  # Number of logs to generate
    logs = generate_logs(num_logs)
    
    # Save logs to file
    save_logs_to_file(logs)
    print(f"Logs saved to logs.json")
    
    # Save logs to DataFrame
    df = save_logs_to_dataframe(logs)
    print(f"Logs saved to logs.csv")
    print(df.head())  # Display the first few logs
