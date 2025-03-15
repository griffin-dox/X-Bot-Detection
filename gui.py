import streamlit as st
import joblib
import numpy as np
import re
import snscrape.modules.twitter as sntwitter

def load_models():
    """Load models and preprocessing objects."""
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    scaler = joblib.load("models/scaler.pkl")
    lgbm_model = joblib.load("models/lgbm_model.pkl")
    return vectorizer, scaler, lgbm_model

def extract_username(profile_link):
    """Extract username from an X profile link."""
    match = re.search(r"x\.com/([a-zA-Z0-9_]+)", profile_link)
    if not match:
        st.error("Invalid X profile link. Please enter a valid link.")
        return None
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
        st.error(f"Error scraping profile: {str(e)}")
        return None, None

def preprocess_data(profile_data, tweets, vectorizer, scaler):
    """Preprocess scraped data for prediction."""
    all_text = profile_data["bio"] + " " + " ".join(tweets)
    text_features = vectorizer.transform([all_text]).toarray()
    numerical_features = np.array([
        profile_data["followers_count"],
        profile_data["following_count"],
        profile_data["tweet_count"],
        profile_data["account_age_days"]
    ]).reshape(1, -1)
    scaled_numerical_features = scaler.transform(numerical_features)
    return np.hstack([text_features, scaled_numerical_features])

def predict_bot(features, lgbm_model):
    """Make a bot prediction using the trained model."""
    prediction = lgbm_model.predict(features)[0]
    return "Bot" if prediction == 1 else "Human"

def main():
    st.title("Twitter Bot Detection")
    profile_link = st.text_input("Enter X profile link:")
    
    if st.button("Analyze Profile"):
        username = extract_username(profile_link)
        if username:
            vectorizer, scaler, lgbm_model = load_models()
            profile_data, tweets = scrape_profile(username)
            if profile_data and tweets:
                features = preprocess_data(profile_data, tweets, vectorizer, scaler)
                result = predict_bot(features, lgbm_model)
                st.success(f"Prediction: {result}")

if __name__ == "__main__":
    main()
