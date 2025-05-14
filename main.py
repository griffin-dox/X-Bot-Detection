import os
from src.utils import (
    load_models,
    extract_username,
    scrape_profile,
    preprocess_data,
    predict_bot
)

def format_number(num):
    """Format number with commas."""
    return f"{num:,}"

def format_time(seconds):
    """Format time in seconds to human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    elif seconds < 86400:
        return f"{seconds/3600:.1f} hours"
    else:
        return f"{seconds/86400:.1f} days"

def main():
    """Main function to run the bot detection system."""
    try:
        # Load models
        (distilbert_model, tokenizer, fasttext_model, lgbm_model, 
         vectorizer, scaler, anomaly_model, numerical_features) = load_models()
        
        # Get profile link from user
        profile_link = input("Enter X profile link: ")
        
        # Extract username and get profile data
        username = extract_username(profile_link)
        profile_data, tweets = scrape_profile(username)
        
        # Preprocess data and make prediction
        features, engineered_features = preprocess_data(
            profile_data, tweets, vectorizer, scaler, numerical_features
        )
        # Only use LightGBM and anomaly model for prediction
        result = predict_bot(
            features, '', engineered_features,  # Pass empty string for bio
            None, None, None, lgbm_model, anomaly_model  # Only use lgbm_model and anomaly_model
        )
        
        # Display results
        print("\nBot Detection Results:")
        print("-" * 50)
        print(f"Account: @{username}")
        print(f"Display Name: {profile_data['display_name']}")
        print(f"Prediction: {'Bot' if result['is_bot'] else 'Human'}")
        print(f"Confidence: {result['confidence_score']:.2%}")
        print(f"Decision by: {result['decision_by']}")
        print(f"Anomaly Score: {result['anomaly_score']:.2f}")
        
        print("\nModel Scores:")
        for model, score in result['model_scores'].items():
            print(f"- {model}: {score:.2%}")
        
        print("\nProfile Information:")
        print("-" * 50)
        print(f"Bio: {profile_data['bio']}")
        print(f"Location: {profile_data['location'] or 'Not specified'}")
        print(f"Website: {profile_data['website'] or 'Not specified'}")
        print(f"Created: {profile_data['created_at'].strftime('%Y-%m-%d')}")
        print(f"Verified: {'Yes' if profile_data['verified'] else 'No'}")
        
        print("\nAccount Statistics:")
        print("-" * 50)
        print(f"Followers: {format_number(profile_data['followers_count'])}")
        print(f"Following: {format_number(profile_data['following_count'])}")
        print(f"Total Tweets: {format_number(profile_data['tweet_count'])}")
        print(f"Account Age: {profile_data['account_age_days']} days")
        print(f"Favorites: {format_number(profile_data['favourites_count'])}")
        print(f"Listed: {format_number(profile_data['listed_count'])}")
        print(f"Media Count: {format_number(profile_data['media_count'])}")
        
        print("\nRecent Activity Analysis:")
        print("-" * 50)
        print(f"Recent Tweets Analyzed: {profile_data['recent_tweet_count']}")
        print(f"Average Tweet Interval: {format_time(profile_data['avg_tweet_interval'])}")
        print(f"Average Likes: {format_number(profile_data['avg_likes'])}")
        print(f"Average Retweets: {format_number(profile_data['avg_retweets'])}")
        print(f"Average Replies: {format_number(profile_data['avg_replies'])}")
        print(f"Average Quotes: {format_number(profile_data['avg_quotes'])}")
        
        print("\nContent Analysis:")
        print("-" * 50)
        print(f"Retweet Ratio: {profile_data['retweet_ratio']:.2%}")
        print(f"Reply Ratio: {profile_data['reply_ratio']:.2%}")
        print(f"Quote Ratio: {profile_data['quote_ratio']:.2%}")
        print(f"Media Ratio: {profile_data['media_ratio']:.2%}")
        print(f"Hashtag Ratio: {profile_data['hashtag_ratio']:.2%}")
        print(f"Mention Ratio: {profile_data['mention_ratio']:.2%}")
        print(f"URL Ratio: {profile_data['url_ratio']:.2%}")
        
        print("\nProfile Features:")
        print("-" * 50)
        print(f"Default Profile: {'Yes' if profile_data['default_profile'] else 'No'}")
        print(f"Default Image: {'Yes' if profile_data['default_profile_image'] else 'No'}")
        print(f"Has Profile Image: {'Yes' if profile_data['profile_image_url'] else 'No'}")
        print(f"Has Banner: {'Yes' if profile_data['profile_banner_url'] else 'No'}")
        print(f"Has Location: {'Yes' if profile_data['has_location'] else 'No'}")
        print(f"Has Website: {'Yes' if profile_data['has_website'] else 'No'}")
        print(f"Has Bio: {'Yes' if profile_data['has_bio'] else 'No'}")
        
        print("\nText Analysis:")
        print("-" * 50)
        print(f"Bio Length: {profile_data['bio_length']} characters")
        print(f"Name Length: {profile_data['name_length']} characters")
        print(f"Username Length: {profile_data['username_length']} characters")
        print(f"Average Tweet Length: {engineered_features['avg_tweet_length']:.0f} characters")
        
        print("\nRecent Tweets:")
        print("-" * 50)
        for i, tweet in enumerate(tweets, 1):
            print(f"\n{i}. {tweet['content']}")
            print(f"   Created: {tweet['created_at'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Likes: {format_number(tweet['likes'])} | "
                  f"Retweets: {format_number(tweet['retweets'])} | "
                  f"Replies: {format_number(tweet['replies'])} | "
                  f"Quotes: {format_number(tweet['quotes'])}")
            print(f"   Has Media: {'Yes' if tweet['has_media'] else 'No'} | "
                  f"Has Hashtags: {'Yes' if tweet['has_hashtags'] else 'No'} | "
                  f"Has Mentions: {'Yes' if tweet['has_mentions'] else 'No'} | "
                  f"Has URLs: {'Yes' if tweet['has_urls'] else 'No'}")
            
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("Please ensure all model files are present in the models directory.")
    except ValueError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
