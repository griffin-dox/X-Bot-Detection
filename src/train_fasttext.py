import os
import pandas as pd
import re
from fasttext import train_supervised
from sklearn.model_selection import train_test_split  # Add this import

# Step 1: Load Dataset
def load_data(file_path):
    """Load dataset from CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)

# Step 2: Preprocess Text
def preprocess_text(text):
    """Clean and preprocess text."""
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)     # Remove mentions
    text = re.sub(r"#\w+", "", text)     # Remove hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    return text.lower()

def prepare_fasttext_data(data, labels, output_file="fasttext_train.txt"):
    """Prepare FastText training data."""
    with open(output_file, "w", encoding="utf-8") as f:
        for text, label in zip(data, labels):
            f.write(f"_label_{label} {text}\n")

# Step 3: Train FastText Model
def train_fasttext_model(train_file="fasttext_train.txt", output_file="fasttext_model.bin"):
    """Train a FastText model for text classification."""
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    
    model = train_supervised(
        input=train_file,
        epoch=25,
        lr=1.0,
        wordNgrams=2,
        verbose=2
    )
    model.save_model(output_file)
    print(f"FastText model saved to {output_file}")

# Step 4: Evaluate FastText Model
def evaluate_fasttext_model(model, test_file="fasttext_test.txt"):
    """Evaluate the FastText model."""
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    print("Model Evaluation:")
    print(model.test(test_file))

# Main Function
if __name__ == "__main__":
    # Define paths
    file_path = "bot_detection_data.csv"
    train_file = "fasttext_train.txt"
    test_file = "fasttext_test.txt"
    model_output_file = "fasttext_model.bin"
    
    # Step 1: Load Data
    print("Loading data...")
    data = load_data(file_path)
    
    # Step 2: Handle Missing Values
    print("Handling missing values...")
    data.dropna(subset=['Tweet', 'Bot Label'], inplace=True)
    
    # Step 3: Preprocess Text
    print("Preprocessing text...")
    data['cleaned_text'] = data['Tweet'].apply(preprocess_text)
    data['Bot Label'] = data['Bot Label'].astype(str)  # Ensure labels are strings
    
    # Step 4: Split Data into Training and Testing Sets
    print("Splitting data into training and testing sets...")
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Prepare FastText training and testing data
    print("Preparing FastText training data...")
    prepare_fasttext_data(train_data['cleaned_text'], train_data['Bot Label'], train_file)
    prepare_fasttext_data(test_data['cleaned_text'], test_data['Bot Label'], test_file)
    
    # Step 5: Train FastText Model
    print("Training FastText model...")
    train_fasttext_model(train_file, model_output_file)
    
    # Step 6: Evaluate FastText Model
    print("Evaluating FastText model...")
    model = train_supervised(input=train_file)
    evaluate_fasttext_model(model, test_file)
    
    print("Training and evaluation completed successfully!")