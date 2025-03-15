import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from fasttext import load_model
import joblib

# Load the test dataset
def load_test_data(file_path):
    """Load preprocessed test data."""
    data = pd.read_csv(file_path)
    return data

# Load trained models and preprocessing objects
def load_models():
    """Load DistilBERT, FastText, and LightGBM models."""
    # DistilBERT
    distilbert_model = TFDistilBertForSequenceClassification.from_pretrained("bot-detection-system/models/nlp_model")
    tokenizer = DistilBertTokenizer.from_pretrained("bot-detection-system/models/nlp_model")
    
    # FastText
    fasttext_model = load_model("bot-detection-system/models/fasttext/fasttext_model.bin")
    
    # LightGBM
    lgbm_model = joblib.load("bot-detection-system/models/lightgbm/lgbm_model.pkl")
    
    # Preprocessing objects
    vectorizer = joblib.load("bot-detection-system/models/tfidf_vectorizer.pkl")
    scaler = joblib.load("bot-detection-system/models/scaler.pkl")
    
    return distilbert_model, tokenizer, fasttext_model, lgbm_model, vectorizer, scaler

# Generate predictions
def generate_predictions(cleaned_text, features, distilbert_model, tokenizer, fasttext_model, lgbm_model):
    """Generate predictions from all models."""
    # DistilBERT predictions
    inputs = tokenizer(list(cleaned_text), padding=True, truncation=True, max_length=128, return_tensors="tf")
    distilbert_output = distilbert_model(inputs).logits.numpy()
    distilbert_preds = [pred[1] for pred in distilbert_output]  # Probability of being a bot
    
    # FastText predictions
    fasttext_preds = [float(fasttext_model.predict(text)[0][0]) for text in cleaned_text]
    
    # LightGBM predictions
    lgbm_preds = lgbm_model.predict(features)
    
    return distilbert_preds, fasttext_preds, lgbm_preds

# Combine predictions
def combine_predictions(distilbert_preds, fasttext_preds, lgbm_preds):
    """Combine predictions using a weighted average."""
    combined_scores = []
    for db_pred, ft_pred, lgbm_pred in zip(distilbert_preds, fasttext_preds, lgbm_preds):
        confidence_score = 0.5 * db_pred + 0.3 * ft_pred + 0.2 * lgbm_pred
        combined_scores.append(confidence_score)
    return combined_scores

# Evaluate the hybrid model
def evaluate_model(y_true, y_pred_scores, threshold=0.5):
    """Evaluate the hybrid model."""
    # Convert continuous scores to binary predictions
    y_pred_binary = [1 if score >= threshold else 0 for score in y_pred_scores]
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    auc_roc = roc_auc_score(y_true, y_pred_scores)
    
    print(f"Threshold: {threshold}")
    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}, AUC-ROC: {auc_roc}")
    print("Classification Report:\n", classification_report(y_true, y_pred_binary))

# Main function
if __name__ == "__main__":
    # Load test data
    test_data = load_test_data("data/processed/preprocessed_data.csv")
    X_test = test_data.drop(columns=["Bot Label"])
    y_test = test_data["Bot Label"]
    
    # Load models
    distilbert_model, tokenizer, fasttext_model, lgbm_model, vectorizer, scaler = load_models()
    
    # Preprocess features
    text_features = vectorizer.transform(X_test['cleaned_text']).toarray()
    numerical_features = scaler.transform(X_test[['Retweet Count', 'Mention Count']])
    features = np.hstack([text_features, numerical_features])
    
    # Generate predictions
    distilbert_preds, fasttext_preds, lgbm_preds = generate_predictions(
        X_test['cleaned_text'], features, distilbert_model, tokenizer, fasttext_model, lgbm_model
    )
    
    # Combine predictions
    combined_scores = combine_predictions(distilbert_preds, fasttext_preds, lgbm_preds)
    
    # Evaluate the hybrid model
    evaluate_model(y_test, combined_scores, threshold=0.5)