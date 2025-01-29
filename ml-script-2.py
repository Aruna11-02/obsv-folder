import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def classify_log_level(log_level):
    if log_level == 'ERROR':
        return 1
    elif log_level == 'WARNING':
        return 0.5
    else:  # INFO and DEBUG
        return 0

def find_motifs_in_errors(log_values, window_size=3):
    n = len(log_values)
    if n < window_size:
        return []
    
    distances = []
    for i in range(n - window_size + 1):
        subseq_i = log_values[i:i + window_size]
        for j in range(i + 1, n - window_size + 1):
            subseq_j = log_values[j:j + window_size]
            distance = euclidean(subseq_i, subseq_j)
            distances.append((distance, i, j))
    
    distances = sorted(distances, key=lambda x: x[0])
    motifs = [(i, j) for _, i, j in distances[:3]]
    return motifs

def prepare_and_train_model(file_path):
    # Load and prepare the data
    data = pd.read_csv(file_path)
    data['log_value'] = data['log_level'].apply(classify_log_level)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data_sorted = data.sort_values(by='timestamp')
    
    # Feature Engineering
    data['time_since_last_error'] = data['timestamp'].diff().dt.total_seconds().fillna(0)
    data['rolling_error_count'] = (data['log_value'] == 1).astype(int).rolling(5, min_periods=1).sum()
    data['rolling_warning_count'] = (data['log_value'] == 0.5).astype(int).rolling(5, min_periods=1).sum()
    data['error_after_warning'] = ((data['log_value'] == 1) & (data['log_value'].shift(1) == 0.5)).astype(int)
    
    # Find motifs for the entire dataset
    log_values = data['log_value'].values
    motifs = find_motifs_in_errors(log_values, window_size=3)
    data['motif_count'] = len(motifs)
    
    # Prepare features and target
    features = ['time_since_last_error', 'rolling_error_count', 'rolling_warning_count', 
               'error_after_warning', 'motif_count']
    target = 'log_value'
    
    X = data[features]
    y = (data[target] == 1).astype(int)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train the model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, features

def save_model(model, output_path='log_prediction_model.pkl'):
    """Save the trained model with proper serialization"""
    try:
        joblib.dump(model, output_path)
        print(f"Model successfully saved to {output_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    # Train and save the model
    
    trained_model, model_features = prepare_and_train_model('rachit-1.csv')
    
    # Verify model has required attributes before saving
    if hasattr(trained_model, 'predict') and hasattr(trained_model, 'predict_proba'):
        # Save model with feature information
        model_data = {
            'model': trained_model,
            'features': model_features,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        save_model(model_data)
        
        # Verify the saved model
        try:
            loaded_model = joblib.load('log_prediction_model.pkl')
            print("\nModel verification:")
            print("- Model loaded successfully")
            print("- Contains predict method:", hasattr(loaded_model['model'], 'predict'))
            print("- Contains features list:", 'features' in loaded_model)
            print("- Training timestamp:", loaded_model['timestamp'])
        except Exception as e:
            print(f"Error during model verification: {e}")
    else:
        print("Error: Model missing required methods 'predict' or 'predict_proba'")
