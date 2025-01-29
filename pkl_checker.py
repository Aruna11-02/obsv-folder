import joblib

model = joblib.load('model_and_features.pkl')
print(type(model))  # Check the type of the loaded object
