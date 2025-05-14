import argparse
from src.utils import (
    load_models,
    extract_username,
    scrape_profile,
    preprocess_data,
    predict_bot
)

def main():
    """CLI interface for the bot detection system."""
    parser = argparse.ArgumentParser(description="Twitter Bot Detector CLI")
    parser.add_argument("profile_link", type=str, help="X profile link")
    args = parser.parse_args()
    
    try:
        # Load models
        distilbert_model, tokenizer, fasttext_model, lgbm_model, vectorizer, scaler = load_models()
        
        # Extract username and get profile data
        username = extract_username(args.profile_link)
        profile_data, tweets = scrape_profile(username)
        
        # Preprocess data and make prediction
        features = preprocess_data(profile_data, tweets, vectorizer, scaler)
        result = predict_bot(features, profile_data["bio"], distilbert_model, tokenizer, fasttext_model, lgbm_model)
        
        # Display results
        print("\nBot Detection Results:")
        print("-" * 50)
        print(f"Account: @{username}")
        print(f"Prediction: {'Bot' if result['is_bot'] else 'Human'}")
        print(f"Confidence: {result['confidence_score']:.2%}")
        print(f"Decision by: {result['decision_by']}")
        print("\nModel Scores:")
        for model, score in result['model_scores'].items():
            print(f"- {model}: {score:.2%}")
            
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("Please ensure all model files are present in the models directory.")
    except ValueError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()