# Utility Functions

import os
import re
import numpy as np
import pickle
import tensorflow as tf
import requests
from bs4 import BeautifulSoup
from fasttext import load_model
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
from datetime import datetime
import time
import joblib
from requests.exceptions import SSLError
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, WebDriverException

# Suppress scikit-learn version mismatch warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Import transformers
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

# Nitter instances (add more if needed)
NITTER_INSTANCES = [
    'https://nitter.net',
    'https://nitter.1d4.us',
    'https://nitter.kavin.rocks',
    'https://nitter.unixfox.eu',
    'https://nitter.poast.org',           # Singapore/Asia
    'https://nitter.privacydev.net',      # Global
    'https://nitter.catsarch.com',        # Global
    'https://nitter.42l.fr',              # Europe, but often fast
    'https://nitter.moomoo.me',           # Global, moved to end
]

def get_working_nitter_instance():
    """Get a working Nitter instance."""
    for instance in NITTER_INSTANCES:
        try:
            response = requests.get(instance, timeout=5)
            if response.status_code == 200:
                return instance
        except:
            continue
    raise RuntimeError("No working Nitter instances found")

def get_model_paths():
    """Get paths to all model files."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return {
        "distilbert": os.path.join(base_dir, "models", "distilbert_model"),
        "fasttext": os.path.join(base_dir, "models", "fasttext", "fasttext_model.bin"),
        "lgbm": os.path.join(base_dir, "models", "lightgbm", "lgbm_model.pkl"),
        "vectorizer": os.path.join(base_dir, "models", "tfidf_vectorizer.pkl"),
        "scaler": os.path.join(base_dir, "models", "scaler.pkl"),
        "anomaly_model": os.path.join(base_dir, "models", "anomaly_model.pkl"),
        "numerical_features": os.path.join(base_dir, "models", "numerical_features.pkl")
    }

def validate_model_files():
    """Validate that all required model files exist."""
    paths = get_model_paths()
    missing_files = []
    for name, path in paths.items():
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
    if missing_files:
        raise FileNotFoundError(
            "Missing model files:\n" + "\n".join(missing_files)
        )

def try_load(path):
    # Try joblib
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"joblib.load failed for {path}: {e}")
    # Try pickle
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"pickle.load failed for {path}: {e}")
    # Try pandas
    try:
        return pd.read_pickle(path)
    except Exception as e:
        print(f"pd.read_pickle failed for {path}: {e}")
    print(f"All loading methods failed for {path}")
    return None

def load_models():
    """Load all models and preprocessing objects."""
    validate_model_files()
    paths = get_model_paths()
    try:
        distilbert_model = TFDistilBertForSequenceClassification.from_pretrained(paths["distilbert"])
        tokenizer = DistilBertTokenizer.from_pretrained(paths["distilbert"])
        try:
            fasttext_model = load_model(paths["fasttext"])
        except Exception as e:
            print(f"Warning: Error loading FastText model: {str(e)}")
            fasttext_model = None
        try:
            lgbm_model = try_load(paths["lgbm"])
        except Exception as e:
            print(f"Warning: Error loading LightGBM model: {str(e)}")
            lgbm_model = None
        try:
            vectorizer = try_load(paths["vectorizer"])
        except Exception as e:
            print(f"Warning: Error loading vectorizer: {str(e)}")
            vectorizer = None
        try:
            scaler = try_load(paths["scaler"])
        except Exception as e:
            print(f"Warning: Error loading scaler: {str(e)}")
            scaler = None
        try:
            anomaly_model = try_load(paths["anomaly_model"])
        except Exception as e:
            print(f"Warning: Error loading anomaly model: {str(e)}")
            anomaly_model = None
        try:
            numerical_features = try_load(paths["numerical_features"])
        except Exception as e:
            print(f"Warning: Error loading numerical features: {str(e)}")
            numerical_features = None
        return (distilbert_model, tokenizer, fasttext_model, lgbm_model, 
                vectorizer, scaler, anomaly_model, numerical_features)
    except Exception as e:
        raise RuntimeError(f"Error loading models: {str(e)}")

def extract_username(profile_link):
    """Extract username from an X profile link."""
    match = re.search(r"x\.com/([a-zA-Z0-9_]+)", profile_link)
    if not match:
        raise ValueError("Invalid X profile link. Please enter a valid link.")
    return match.group(1)

def safe_get_text(soup, selector, default=""):
    el = soup.select_one(selector)
    return el.text.strip() if el else default

def safe_get_attr(soup, selector, attr, default=None):
    el = soup.select_one(selector)
    return el[attr] if el and el.has_attr(attr) else default

def scrape_profile(username):
    """Scrape X profile and tweets using Selenium headless browser only."""
    options = Options()
    options.headless = True
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--window-size=1920,1080')
    driver = None
    try:
        driver = webdriver.Chrome(options=options)
        url = f'https://x.com/{username}'
        driver.get(url)
        driver.implicitly_wait(10)
        # Extract display name
        try:
            display_name = driver.find_element(By.XPATH, '//div[@data-testid="UserName"]//span').text
        except NoSuchElementException:
            display_name = username
        # Extract bio
        try:
            bio = driver.find_element(By.XPATH, '//div[@data-testid="UserDescription"]').text
        except NoSuchElementException:
            bio = ''
        # Extract location
        try:
            location = driver.find_element(By.XPATH, '//span[@data-testid="UserLocation"]').text
        except NoSuchElementException:
            location = ''
        # Extract stats
        try:
            stats = driver.find_elements(By.XPATH, '//div[@data-testid="UserProfileHeader_Items"]/span')
            website = ''
            for stat in stats:
                if stat.text.startswith('http'):
                    website = stat.text
        except NoSuchElementException:
            website = ''
        # Extract join date
        try:
            join_date = driver.find_element(By.XPATH, '//span[contains(text(),"Joined")]').text.replace('Joined ', '')
        except NoSuchElementException:
            join_date = ''
        # Extract followers, following, tweets
        try:
            followers = driver.find_element(By.XPATH, '//a[contains(@href,"/followers")]/span[1]/span').text.replace(',', '')
        except NoSuchElementException:
            followers = '0'
        try:
            following = driver.find_element(By.XPATH, '//a[contains(@href,"/following")]/span[1]/span').text.replace(',', '')
        except NoSuchElementException:
            following = '0'
        try:
            tweet_count = driver.find_element(By.XPATH, '//div[@data-testid="primaryColumn"]//span[contains(text()," Tweets")]').text.split(' ')[0].replace(',', '')
        except NoSuchElementException:
            tweet_count = '0'
        # Extract profile image
        try:
            profile_image_url = driver.find_element(By.XPATH, '//img[@alt="Image"]').get_attribute('src')
        except NoSuchElementException:
            profile_image_url = None
        # Extract recent tweets
        tweets_data = []
        tweet_elements = driver.find_elements(By.XPATH, '//div[@data-testid="tweet"]')
        for tweet in tweet_elements[:50]:
            try:
                content = tweet.find_element(By.XPATH, './/div[2]//div[2]//div[1]').text
            except NoSuchElementException:
                content = ''
            try:
                created_at = tweet.find_element(By.XPATH, './/time').get_attribute('datetime')
            except NoSuchElementException:
                created_at = ''
            try:
                likes = tweet.find_element(By.XPATH, './/div[@data-testid="like"]//span').text.replace(',', '')
            except NoSuchElementException:
                likes = '0'
            try:
                retweets = tweet.find_element(By.XPATH, './/div[@data-testid="retweet"]//span').text.replace(',', '')
            except NoSuchElementException:
                retweets = '0'
            try:
                replies = tweet.find_element(By.XPATH, './/div[@data-testid="reply"]//span').text.replace(',', '')
            except NoSuchElementException:
                replies = '0'
            tweets_data.append({
                'content': content,
                'created_at': created_at,
                'likes': int(likes or '0'),
                'retweets': int(retweets or '0'),
                'replies': int(replies or '0'),
                'quotes': 0,
                'has_media': False,
                'has_hashtags': False,
                'has_mentions': False,
                'has_urls': False
            })
        # Compose profile_data
        from datetime import datetime as dt
        try:
            created_at_dt = dt.strptime(join_date, '%B %Y') if join_date else dt.now()
        except Exception:
            created_at_dt = dt.now()
        profile_data = {
            'username': username,
            'display_name': display_name,
            'bio': bio,
            'location': location,
            'website': website,
            'created_at': created_at_dt,
            'followers_count': int(followers or '0'),
            'following_count': int(following or '0'),
            'tweet_count': int(tweet_count or '0'),
            'account_age_days': (dt.now() - created_at_dt).days,
            'verified': False,  # Not available without login
            'default_profile': False,
            'default_profile_image': False,
            'profile_image_url': profile_image_url,
            'profile_banner_url': None,
            'favourites_count': 0,
            'listed_count': 0,
            'media_count': 0,
        }
        return profile_data, tweets_data
    except WebDriverException as e:
        raise RuntimeError(f'Selenium error: {e}')
    finally:
        if driver:
            driver.quit()

def engineer_features(profile_data, tweets, numerical_features):
    """Engineer additional features based on reference data, skipping missing data."""
    # Basic features
    features = {
        "followers_count": profile_data.get("followers_count", 0),
        "following_count": profile_data.get("following_count", 0),
        "tweet_count": profile_data.get("tweet_count", 0),
        "account_age_days": profile_data.get("account_age_days", 0),
        "favourites_count": profile_data.get("favourites_count", 0),
        "listed_count": profile_data.get("listed_count", 0),
        "media_count": profile_data.get("media_count", 0)
    }
    # Calculate ratios
    features["followers_following_ratio"] = (
        features["followers_count"] / (features["following_count"] + 1)
    )
    features["tweets_per_day"] = (
        features["tweet_count"] / (features["account_age_days"] + 1)
    )
    features["media_ratio"] = (
        features["media_count"] / (features["tweet_count"] + 1)
    )
    # Calculate engagement metrics
    features["engagement_rate"] = (
        (features["favourites_count"] + features["listed_count"]) /
        (features["followers_count"] + 1)
    )
    # Add tweet analysis features (use .get with default 0)
    features["retweet_ratio"] = profile_data.get("retweet_ratio", 0)
    features["reply_ratio"] = profile_data.get("reply_ratio", 0)
    features["quote_ratio"] = profile_data.get("quote_ratio", 0)
    features["hashtag_ratio"] = profile_data.get("hashtag_ratio", 0)
    features["mention_ratio"] = profile_data.get("mention_ratio", 0)
    features["url_ratio"] = profile_data.get("url_ratio", 0)
    # Add engagement metrics
    features["avg_likes"] = profile_data.get("avg_likes", 0)
    features["avg_retweets"] = profile_data.get("avg_retweets", 0)
    features["avg_replies"] = profile_data.get("avg_replies", 0)
    features["avg_quotes"] = profile_data.get("avg_quotes", 0)
    features["avg_tweet_interval"] = profile_data.get("avg_tweet_interval", 0)
    # Calculate text-based features
    all_text = profile_data.get("bio", "") + " " + " ".join(tweet.get("content", "") for tweet in tweets)
    features["avg_tweet_length"] = (
        np.mean([len(tweet.get("content", "")) for tweet in tweets]) if tweets else 0
    )
    features["bio_length"] = profile_data.get("bio_length", 0)
    features["name_length"] = profile_data.get("name_length", 0)
    features["username_length"] = profile_data.get("username_length", 0)
    features["total_text_length"] = len(all_text)
    # Add boolean features
    features["is_verified"] = int(profile_data.get("verified", False))
    features["has_default_profile"] = int(profile_data.get("default_profile", False))
    features["has_default_image"] = int(profile_data.get("default_profile_image", False))
    features["has_profile_image"] = int(profile_data.get("profile_image_url") is not None)
    features["has_banner"] = int(profile_data.get("profile_banner_url") is not None)
    features["has_location"] = int(profile_data.get("location", "") != "")
    features["has_website"] = int(profile_data.get("website", "") != "")
    features["has_bio"] = int(profile_data.get("bio", "") != "")
    return features

def preprocess_data(profile_data, tweets, vectorizer, scaler, numerical_features):
    """Preprocess scraped data for prediction (numerical features only)."""
    engineered_features = engineer_features(profile_data, tweets, numerical_features)
    if hasattr(numerical_features, 'columns'):
        feature_names = list(numerical_features.columns)
    else:
        n_expected = scaler.mean_.shape[0]
        feature_names = list(engineered_features.keys())[:n_expected]
    numerical_array = np.array([engineered_features[col] for col in feature_names]).reshape(1, -1)
    print("Feature names:", feature_names)
    print("Numerical array shape:", numerical_array.shape)
    print("Scaler expects:", scaler.mean_.shape)
    scaled_numerical_features = scaler.transform(numerical_array)
    # Skip text features and vectorizer
    return scaled_numerical_features, engineered_features

def predict_bot(features, bio, engineered_features, distilbert_model, tokenizer, 
                fasttext_model, lgbm_model, anomaly_model):
    """Make a bot prediction using only numerical models (LightGBM, anomaly)."""
    # Get LightGBM prediction
    lgbm_pred = lgbm_model.predict(features)[0]
    # Get anomaly score
    anomaly_score = anomaly_model.score_samples(features)[0]
    anomaly_pred = 1 if anomaly_score < -0.5 else 0  # Threshold can be adjusted
    # Confidence-Based Model Selection
    predictions = {
        "LightGBM": lgbm_pred,
        "Anomaly": 1 - (anomaly_score + 0.5)  # Convert to [0,1] range
    }
    # Calculate final prediction
    best_model = max(predictions, key=predictions.get)
    best_confidence = predictions[best_model]
    if best_confidence > 0.85:
        final_prediction = {
            "is_bot": best_confidence >= 0.5,
            "confidence_score": best_confidence,
            "decision_by": best_model,
            "model_scores": predictions,
            "anomaly_score": anomaly_score,
            "engineered_features": engineered_features
        }
    else:
        # Use weighted average of both models
        weights = {
            "LightGBM": 0.7,
            "Anomaly": 0.3
        }
        final_confidence = sum(predictions[model] * weights[model] for model in predictions)
        final_prediction = {
            "is_bot": final_confidence >= 0.5,
            "confidence_score": final_confidence,
            "decision_by": "Weighted Average",
            "model_scores": predictions,
            "anomaly_score": anomaly_score,
            "engineered_features": engineered_features
        }
    return final_prediction