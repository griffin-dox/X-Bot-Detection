import joblib
import numpy as np
import re
import snscrape.modules.twitter as sntwitter
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from fasttext import load_model
import os

def load_models():
    """Load models and preprocessing objects."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    distilbert_path = os.path.join(base_dir, "models", "distilbert_model")
    fasttext_path = os.path.join(base_dir, "models", "fasttext", "fasttext_model.bin")
    lgbm_path = os.path.join(base_dir, "models", "lightgbm", "lgbm_model.pkl")
    vectorizer_path = os.path.join(base_dir, "models", "tfidf_vectorizer.pkl")
    scaler_path = os.path.join(base_dir, "models", "scaler.pkl")
    
    distilbert_model = TFDistilBertForSequenceClassification.from_pretrained(distilbert_path)
    tokenizer = DistilBertTokenizer.from_pretrained(distilbert_path)
    fasttext_model = load_model(fasttext_path)
    lgbm_model = joblib.load(lgbm_path)
    vectorizer = joblib.load(vectorizer_path)
    scaler = joblib.load(scaler_path)
    
    return distilbert_model, tokenizer, fasttext_model, lgbm_model, vectorizer, scaler

# Load models globally
distilbert_model, tokenizer, fasttext_model, lgbm_model, vectorizer, scaler = load_models()

def extract_username(profile_link):
    """Extract username from an X profile link."""
    match = re.search(r"x\.com/([a-zA-Z0-9_]+)", profile_link)
    if not match:
        raise ValueError("Invalid X profile link.")
    return match.group(1)

def scrape_profile(username):
    """Scrape user profile and tweets using snscrape."""
    try:
        user = next(sntwitter.TwitterUserScraper(username).get_items())
        tweets = [tweet.content for tweet in sntwitter.TwitterUserScraper(username).get_items()][:10]
        
        profile_data = {
            "bio": user.rawDescription,
            "followers_count": user.followersCount,
            "following_count": user.friendsCount,
            "tweet_count": user.statusesCount,
            "account_age_days": (np.datetime64('today') - np.datetime64(user.created.strftime('%Y-%m-%d'))).astype(int)
        }
        return profile_data, tweets
    except Exception as e:
        raise Exception(f"Error scraping profile: {str(e)}")

def preprocess_data(profile_data, tweets):
    """Preprocess scraped data for prediction."""
    all_text = profile_data["bio"] + " " + " ".join(tweets)
    text_features = vectorizer.transform([all_text.lower()]).toarray()
    numerical_features = np.array([
        profile_data["followers_count"],
        profile_data["following_count"],
        profile_data["tweet_count"],
        profile_data["account_age_days"]
    ]).reshape(1, -1)
    scaled_numerical_features = scaler.transform(numerical_features)
    return np.hstack([text_features, scaled_numerical_features])

def predict_bot(features, bio):
    """Make a bot prediction using the trained models."""
    inputs = tokenizer(bio, padding=True, truncation=True, max_length=128, return_tensors="tf")
    distilbert_pred = distilbert_model(inputs).logits.numpy()[0][1]
    fasttext_pred = float(fasttext_model.predict(bio)[0][0])
    lgbm_pred = lgbm_model.predict(features)[0]
    
    # Confidence-Based Model Selection
    predictions = {
        "DistilBERT": distilbert_pred,
        "FastText": fasttext_pred,
        "LightGBM": lgbm_pred
    }
    best_model = max(predictions, key=predictions.get)
    best_confidence = predictions[best_model]
    
    if best_confidence > 0.85:
        return {"is_bot": best_confidence >= 0.5, "confidence_score": best_confidence, "decision_by": best_model}
    else:
        final_confidence = (distilbert_pred + fasttext_pred + lgbm_pred) / 3
        return {"is_bot": final_confidence >= 0.5, "confidence_score": final_confidence, "decision_by": "Average"}
