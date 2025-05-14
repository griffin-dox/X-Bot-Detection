import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Path to your training data CSV
TRAIN_DATA_PATH = os.path.join('data', 'processed', 'train_data.csv')
# Path to save the fitted vectorizer
VECTORIZER_PATH = os.path.join('models', 'tfidf_vectorizer.pkl')

# Load training data (adjust column name if needed)
df = pd.read_csv(TRAIN_DATA_PATH)

# Try to find a text column (commonly 'text', 'tweet', or 'content')
text_col = None
for col in df.columns:
    if col.lower() in ['text', 'tweet', 'content', 'cleaned_text', 'bio']:
        text_col = col
        break
if text_col is None:
    raise ValueError('No suitable text column found in training data.')

texts = df[text_col].astype(str).tolist()

# Fit the TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
vectorizer.fit(texts)

# Save the fitted vectorizer
joblib.dump(vectorizer, VECTORIZER_PATH)
print(f'TF-IDF vectorizer fitted and saved to {VECTORIZER_PATH}')
