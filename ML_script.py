import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from scipy.sparse import hstack

# Step 1: Load the log data
file_path = 'logs.csv'
logs_df = pd.read_csv(file_path)

# Step 2: Preprocess the data
log_messages = logs_df['message'].tolist()
error_indices = logs_df[logs_df['log_level'] == 'ERROR'].index

# Step 3: Extract sequences of messages leading up to each error
window_size = 5
sequence_data = []
labels = []

for idx in range(len(log_messages) - window_size):
    window = log_messages[idx:idx + window_size]
    is_error = 1 if (idx + window_size) in error_indices else 0
    sequence_data.append(" ".join(window))
    labels.append(is_error)

# Step 4: Identify motifs using n-gram frequency analysis
vectorizer = CountVectorizer(ngram_range=(2, 3), min_df=2)
X_counts = vectorizer.fit_transform(sequence_data)
sum_counts = X_counts.sum(axis=0)
motif_indices = sum_counts.argsort().tolist()[0][-10:]
motifs = [vectorizer.get_feature_names_out()[i] for i in motif_indices]

print("Top 10 Motifs Detected:")
for motif in motifs:
    print(motif)

# Step 5: Use motifs as features in the classification model
motif_vectorizer = CountVectorizer(vocabulary=motifs)
X_motifs = motif_vectorizer.transform(sequence_data)
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(sequence_data)
X_combined = hstack([X_tfidf, X_motifs])

# Step 6: Handle class imbalance with SMOTE
y = labels
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_combined, y)

# Step 7: Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42
)

# Step 8: Train an XGBoost classifier
model = XGBClassifier(
    max_depth=7,
    learning_rate=0.05,
    n_estimators=300,
    scale_pos_weight=1,
    
    eval_metric='logloss',
    n_jobs=-1
)
model.fit(X_train, y_train)

# Step 9: Evaluate the model
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 10: Display important features
feature_importances = model.feature_importances_
feature_names = tfidf_vectorizer.get_feature_names_out().tolist() + motifs
important_features = sorted(
    zip(feature_names, feature_importances), key=lambda x: -x[1]
)
print("\nTop 10 Important Features:")
for feature, importance in important_features[:10]:
    print(f"{feature}: {importance:.4f}")

# Step 11: Save everything into a single pickle file
to_save = {
    "model": model,
    "tfidf_vectorizer": tfidf_vectorizer,
    "motif_vectorizer": motif_vectorizer,
    "important_features": important_features,
    "motifs": motifs,
}

with open("model_and_features.pkl", "wb") as file:
    pickle.dump(to_save, file)

print("\nModel, vectorizers, and features saved into 'model_and_features.pkl'.")