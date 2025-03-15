import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Step 1: Load Dataset
def load_data(file_path):
    """Load dataset from CSV."""
    try:
        data = pd.read_csv(file_path)
        logging.info("Dataset loaded successfully.")
        return data
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

# Step 2: Preprocess Numerical Features
def preprocess_numerical_features(data):
    """Normalize numerical features."""
    try:
        # Check if required columns exist
        required_columns = ['Retweet Count', 'Mention Count', 'Bot Label']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Dataset must contain the following columns: {required_columns}")
        
        # Handle missing values
        data = data.dropna(subset=required_columns)
        
        # Normalize numerical features
        scaler = StandardScaler()
        numerical_features = scaler.fit_transform(data[['Retweet Count', 'Mention Count']])
        labels = data['Bot Label'].values
        logging.info("Numerical features preprocessed successfully.")
        return numerical_features, labels, scaler
    except Exception as e:
        logging.error(f"Error preprocessing numerical features: {e}")
        raise

# Step 3: Train LightGBM Model
def train_lightgbm(features, labels, scaler, output_dir="bot-detection-system/models"):
    """Train a LightGBM model for final classification."""
    try:
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Define parameters
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        # Train the model
        model = lgb.train(params, train_data, num_boost_round=100, valid_sets=[test_data], early_stopping_rounds=10)
        
        # Evaluate the model
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        auc_score = roc_auc_score(y_test, y_pred)
        logging.info(f"Test AUC Score: {auc_score:.4f}")
        
        # Save the model and scaler
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(model, os.path.join(output_dir, "lgbm_model.pkl"))
        joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
        logging.info(f"LightGBM model and scaler saved to {output_dir}")
    except Exception as e:
        logging.error(f"Error training LightGBM model: {e}")
        raise

# Main Function
if __name__ == "__main__":
    # Define paths
    file_path = os.path.join("bot-detection-system", "data", "raw", "twitter_bot_dataset.csv")
    
    # Step 1: Load Data
    data = load_data(file_path)
    
    # Step 2: Preprocess Numerical Features
    numerical_features, labels, scaler = preprocess_numerical_features(data)
    
    # Step 3: Train LightGBM Model
    train_lightgbm(numerical_features, labels, scaler)