if __name__ == "__main__":
    # Define paths
    file_path = "bot_detection_data.csv"
    output_dir = "models"
    
    # Step 1: Load Data
    data = load_data(file_path)
    
    # Step 2: Perform EDA
    perform_eda(data)
    
    # Step 3: Preprocess Data
    features, labels, vectorizer, scaler, cleaned_text = preprocess_data(data)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features,  # Use preprocessed features
        labels,
        test_size=0.2,
        random_state=42
    )
    
    # Step 4: Train NLP Model
    nlp_model, tokenizer = train_nlp_model(cleaned_text, labels)
    
    # Step 5: Train Anomaly Detection Model
    print("Training Anomaly Detection Model...")
    anomaly_predictions, anomaly_model = train_anomaly_model(X_train)
    
    # Step 8: Save Models
    print("Saving Models...")
    save_models(vectorizer, scaler, nlp_model, anomaly_model, output_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(os.path.join(output_dir, "nlp_tokenizer"))
    
    # Step 6: Combine Models
    print("Combining Models...")
    nlp_inputs = tokenizer(list(cleaned_text), padding=True, truncation=True, max_length=128, return_tensors="tf")
    nlp_outputs = nlp_model(nlp_inputs).logits.numpy()  # Get logits from the model
    nlp_predictions = [1 if pred[0] > pred[1] else 0 for pred in nlp_outputs]  # Convert logits to binary predictions
    combined_scores = combine_models(nlp_predictions, anomaly_predictions)
    
    # Step 7: Evaluate Model
    print("Evaluating Model...")
    thresholds = [0.3, 0.4, 0.5]  # Experiment with different thresholds
    for threshold in thresholds:
        print(f"Evaluation with Threshold={threshold}:")
        evaluate_model(y_test, combined_scores, threshold=threshold)  # Use test data for evaluation
    
    print("Training completed successfully!")