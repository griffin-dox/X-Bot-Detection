import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
from src.utils import (
    load_models,
    extract_username,
    scrape_profile,
    preprocess_data,
    predict_bot
)

class BotDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("X Bot Detection")
        self.root.geometry("1200x800")
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create input section
        self.create_input_section()
        
        # Create results section
        self.create_results_section()
        
        # Load models in background
        self.models = None
        threading.Thread(target=self.load_models_thread, daemon=True).start()
    
    def create_input_section(self):
        """Create the input section of the GUI."""
        input_frame = ttk.LabelFrame(self.main_frame, text="Input", padding="5")
        input_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(input_frame, text="X Profile Link:").grid(row=0, column=0, sticky=tk.W)
        self.profile_link = ttk.Entry(input_frame, width=50)
        self.profile_link.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        
        self.analyze_button = ttk.Button(input_frame, text="Analyze", command=self.analyze_profile)
        self.analyze_button.grid(row=0, column=2, padx=5)
        
        self.status_label = ttk.Label(input_frame, text="Loading models...")
        self.status_label.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=5)
    
    def create_results_section(self):
        """Create the results section of the GUI."""
        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Summary tab
        self.summary_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.summary_tab, text="Summary")
        
        # Profile tab
        self.profile_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.profile_tab, text="Profile")
        
        # Activity tab
        self.activity_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.activity_tab, text="Activity")
        
        # Content tab
        self.content_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.content_tab, text="Content")
        
        # Tweets tab
        self.tweets_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.tweets_tab, text="Recent Tweets")
        
        # Create content for each tab
        self.create_summary_tab()
        self.create_profile_tab()
        self.create_activity_tab()
        self.create_content_tab()
        self.create_tweets_tab()
    
    def create_summary_tab(self):
        """Create the summary tab content."""
        frame = ttk.Frame(self.summary_tab, padding="5")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Prediction section
        pred_frame = ttk.LabelFrame(frame, text="Bot Detection Results", padding="5")
        pred_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.prediction_label = ttk.Label(pred_frame, text="")
        self.prediction_label.grid(row=0, column=0, sticky=tk.W)
        
        self.confidence_label = ttk.Label(pred_frame, text="")
        self.confidence_label.grid(row=1, column=0, sticky=tk.W)
        
        self.decision_label = ttk.Label(pred_frame, text="")
        self.decision_label.grid(row=2, column=0, sticky=tk.W)
        
        self.anomaly_label = ttk.Label(pred_frame, text="")
        self.anomaly_label.grid(row=3, column=0, sticky=tk.W)
        
        # Model scores section
        scores_frame = ttk.LabelFrame(frame, text="Model Scores", padding="5")
        scores_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.scores_text = scrolledtext.ScrolledText(scores_frame, height=5, width=50)
        self.scores_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
    
    def create_profile_tab(self):
        """Create the profile tab content."""
        frame = ttk.Frame(self.profile_tab, padding="5")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Basic info section
        basic_frame = ttk.LabelFrame(frame, text="Basic Information", padding="5")
        basic_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.basic_info_text = scrolledtext.ScrolledText(basic_frame, height=10, width=50)
        self.basic_info_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Account stats section
        stats_frame = ttk.LabelFrame(frame, text="Account Statistics", padding="5")
        stats_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.stats_text = scrolledtext.ScrolledText(stats_frame, height=10, width=50)
        self.stats_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
    
    def create_activity_tab(self):
        """Create the activity tab content."""
        frame = ttk.Frame(self.activity_tab, padding="5")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Recent activity section
        activity_frame = ttk.LabelFrame(frame, text="Recent Activity", padding="5")
        activity_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.activity_text = scrolledtext.ScrolledText(activity_frame, height=10, width=50)
        self.activity_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Engagement metrics section
        engagement_frame = ttk.LabelFrame(frame, text="Engagement Metrics", padding="5")
        engagement_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.engagement_text = scrolledtext.ScrolledText(engagement_frame, height=10, width=50)
        self.engagement_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
    
    def create_content_tab(self):
        """Create the content tab content."""
        frame = ttk.Frame(self.content_tab, padding="5")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Content analysis section
        content_frame = ttk.LabelFrame(frame, text="Content Analysis", padding="5")
        content_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.content_text = scrolledtext.ScrolledText(content_frame, height=10, width=50)
        self.content_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Text analysis section
        text_frame = ttk.LabelFrame(frame, text="Text Analysis", padding="5")
        text_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.text_text = scrolledtext.ScrolledText(text_frame, height=10, width=50)
        self.text_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
    
    def create_tweets_tab(self):
        """Create the tweets tab content."""
        frame = ttk.Frame(self.tweets_tab, padding="5")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.tweets_text = scrolledtext.ScrolledText(frame, height=30, width=100)
        self.tweets_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    def load_models_thread(self):
        """Load models in background thread."""
        try:
            self.models = load_models()
            self.status_label.config(text="Ready to analyze")
            self.analyze_button.config(state="normal")
    except Exception as e:
            self.status_label.config(text=f"Error loading models: {str(e)}")
    
    def format_number(self, num):
        """Format number with commas."""
        return f"{num:,}"
    
    def format_time(self, seconds):
        """Format time in seconds to human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            return f"{seconds/60:.1f} minutes"
        elif seconds < 86400:
            return f"{seconds/3600:.1f} hours"
        else:
            return f"{seconds/86400:.1f} days"
    
    def analyze_profile(self):
        """Analyze the provided X profile."""
        if not self.models:
            self.status_label.config(text="Models not loaded yet")
            return
        
        try:
            self.status_label.config(text="Analyzing profile...")
            self.analyze_button.config(state="disabled")
            
            # Get profile data
            profile_link = self.profile_link.get()
        username = extract_username(profile_link)
            profile_data, tweets = scrape_profile(username)
            
            # Preprocess data and make prediction
            features, engineered_features = preprocess_data(
                profile_data, tweets, self.models[3], self.models[4], self.models[7]
            )
            result = predict_bot(
                features, profile_data["bio"], engineered_features,
                self.models[0], self.models[1], self.models[2], self.models[5], self.models[6]
            )
            
            # Update summary tab
            self.prediction_label.config(
                text=f"Prediction: {'Bot' if result['is_bot'] else 'Human'}"
            )
            self.confidence_label.config(
                text=f"Confidence: {result['confidence_score']:.2%}"
            )
            self.decision_label.config(
                text=f"Decision by: {result['decision_by']}"
            )
            self.anomaly_label.config(
                text=f"Anomaly Score: {result['anomaly_score']:.2f}"
            )
            
            # Update model scores
            self.scores_text.delete(1.0, tk.END)
            for model, score in result['model_scores'].items():
                self.scores_text.insert(tk.END, f"{model}: {score:.2%}\n")
            
            # Update profile tab
            self.basic_info_text.delete(1.0, tk.END)
            self.basic_info_text.insert(tk.END, f"Username: @{username}\n")
            self.basic_info_text.insert(tk.END, f"Display Name: {profile_data['display_name']}\n")
            self.basic_info_text.insert(tk.END, f"Bio: {profile_data['bio']}\n")
            self.basic_info_text.insert(tk.END, f"Location: {profile_data['location'] or 'Not specified'}\n")
            self.basic_info_text.insert(tk.END, f"Website: {profile_data['website'] or 'Not specified'}\n")
            self.basic_info_text.insert(tk.END, f"Created: {profile_data['created_at'].strftime('%Y-%m-%d')}\n")
            self.basic_info_text.insert(tk.END, f"Verified: {'Yes' if profile_data['verified'] else 'No'}\n")
            
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, f"Followers: {self.format_number(profile_data['followers_count'])}\n")
            self.stats_text.insert(tk.END, f"Following: {self.format_number(profile_data['following_count'])}\n")
            self.stats_text.insert(tk.END, f"Total Tweets: {self.format_number(profile_data['tweet_count'])}\n")
            self.stats_text.insert(tk.END, f"Account Age: {profile_data['account_age_days']} days\n")
            self.stats_text.insert(tk.END, f"Favorites: {self.format_number(profile_data['favourites_count'])}\n")
            self.stats_text.insert(tk.END, f"Listed: {self.format_number(profile_data['listed_count'])}\n")
            self.stats_text.insert(tk.END, f"Media Count: {self.format_number(profile_data['media_count'])}\n")
            
            # Update activity tab
            self.activity_text.delete(1.0, tk.END)
            self.activity_text.insert(tk.END, f"Recent Tweets Analyzed: {profile_data['recent_tweet_count']}\n")
            self.activity_text.insert(tk.END, f"Average Tweet Interval: {self.format_time(profile_data['avg_tweet_interval'])}\n")
            self.activity_text.insert(tk.END, f"Average Likes: {self.format_number(profile_data['avg_likes'])}\n")
            self.activity_text.insert(tk.END, f"Average Retweets: {self.format_number(profile_data['avg_retweets'])}\n")
            self.activity_text.insert(tk.END, f"Average Replies: {self.format_number(profile_data['avg_replies'])}\n")
            self.activity_text.insert(tk.END, f"Average Quotes: {self.format_number(profile_data['avg_quotes'])}\n")
            
            self.engagement_text.delete(1.0, tk.END)
            self.engagement_text.insert(tk.END, f"Followers/Following Ratio: {engineered_features['followers_following_ratio']:.2f}\n")
            self.engagement_text.insert(tk.END, f"Tweets per Day: {engineered_features['tweets_per_day']:.2f}\n")
            self.engagement_text.insert(tk.END, f"Engagement Rate: {engineered_features['engagement_rate']:.2%}\n")
            
            # Update content tab
            self.content_text.delete(1.0, tk.END)
            self.content_text.insert(tk.END, f"Retweet Ratio: {profile_data['retweet_ratio']:.2%}\n")
            self.content_text.insert(tk.END, f"Reply Ratio: {profile_data['reply_ratio']:.2%}\n")
            self.content_text.insert(tk.END, f"Quote Ratio: {profile_data['quote_ratio']:.2%}\n")
            self.content_text.insert(tk.END, f"Media Ratio: {profile_data['media_ratio']:.2%}\n")
            self.content_text.insert(tk.END, f"Hashtag Ratio: {profile_data['hashtag_ratio']:.2%}\n")
            self.content_text.insert(tk.END, f"Mention Ratio: {profile_data['mention_ratio']:.2%}\n")
            self.content_text.insert(tk.END, f"URL Ratio: {profile_data['url_ratio']:.2%}\n")
            
            self.text_text.delete(1.0, tk.END)
            self.text_text.insert(tk.END, f"Bio Length: {profile_data['bio_length']} characters\n")
            self.text_text.insert(tk.END, f"Name Length: {profile_data['name_length']} characters\n")
            self.text_text.insert(tk.END, f"Username Length: {profile_data['username_length']} characters\n")
            self.text_text.insert(tk.END, f"Average Tweet Length: {engineered_features['avg_tweet_length']:.0f} characters\n")
            
            # Update tweets tab
            self.tweets_text.delete(1.0, tk.END)
            for i, tweet in enumerate(tweets, 1):
                self.tweets_text.insert(tk.END, f"\n{i}. {tweet['content']}\n")
                self.tweets_text.insert(tk.END, f"   Created: {tweet['created_at'].strftime('%Y-%m-%d %H:%M:%S')}\n")
                self.tweets_text.insert(tk.END, f"   Likes: {self.format_number(tweet['likes'])} | "
                                               f"Retweets: {self.format_number(tweet['retweets'])} | "
                                               f"Replies: {self.format_number(tweet['replies'])} | "
                                               f"Quotes: {self.format_number(tweet['quotes'])}\n")
                self.tweets_text.insert(tk.END, f"   Has Media: {'Yes' if tweet['has_media'] else 'No'} | "
                                               f"Has Hashtags: {'Yes' if tweet['has_hashtags'] else 'No'} | "
                                               f"Has Mentions: {'Yes' if tweet['has_mentions'] else 'No'} | "
                                               f"Has URLs: {'Yes' if tweet['has_urls'] else 'No'}\n")
            
            self.status_label.config(text="Analysis complete")
            self.analyze_button.config(state="normal")
            
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
            self.analyze_button.config(state="normal")

def main():
    root = tk.Tk()
    app = BotDetectionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
