# Train and save the model
    trained_model, model_features = prepare_and_train_model('Rachit.csv')
    
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
