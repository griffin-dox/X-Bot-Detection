# 🥵 Bot Detector

## 📌 Overview
This project aims to develop a **machine learning-based system** for detecting bots on the social media platform **X (formerly Twitter)**. Bots are automated accounts that mimic human behavior and can be used for malicious purposes, such as spreading misinformation, spamming, or manipulating public opinion. By leveraging numerical features (e.g., retweet counts, mention counts) and advanced machine learning techniques, this system identifies bot accounts with high accuracy.

🚀 Developed by **Team Cyfer-Trace, SSPU**.

---

## ✨ Features
✔️ Detects bots based on **text patterns, posting behavior, and engagement metrics**.
✔️ Provides a **confidence score** indicating the likelihood of an account being a bot.
✔️ Generates structured outputs summarizing **bot classification results**.
✔️ Supports **real-time analysis** of social media profiles using `snscrape`.

---

## 🌝 Acknowledgements
- 📊 [Twitter-Bot Detection Dataset](https://www.kaggle.com/datasets/goyaladi/twitter-bot-detection-dataset)

---

## 👥 Authors
- 🏆 [Team-Cyfer-Trace](https://github.com/Team-Cyfer-Trace)
- 👨‍💻 [Team Lead @griffin-dox](https://github.com/griffin-dox)
- 👨‍💻 [Member @Bryan-b-2006](https://github.com/Bryan-b-2006)
- 👨‍💻 [Member @hrishikesh-hiray](https://github.com/hrishikesh-hiray)
- 👨‍💻 [Member @reebharate](https://github.com/reebharate)

---

## ⚙️ Installation

### 📌 Prerequisites
- 🐖 Python **3.8+**
- 📦 Required libraries: `snscrape`, `pandas`, `numpy`, `scikit-learn`, `transformers`, `fasttext`, `joblib`

### 🚀 Steps to Set Up
1️⃣ **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/bot-detection-system.git
   cd bot-detection-system
   ```
2️⃣ **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3️⃣ **Download pre-trained models and preprocessing objects:**
   Place the following files in the `models/` directory:
   - 👤 `distilbert_model/`
   - 📂 `fasttext_model.bin`
   - 📂 `lgbm_model.pkl`
   - 📂 `tfidf_vectorizer.pkl`
   - 📂 `scaler.pkl`

---

## 🚀 Usage

### 🖥️ Data Extraction with `snscrape`
Instead of using the X API, this project utilizes **snscrape** to fetch public data:
```bash
pip install snscrape
```

To scrape tweets from a specific user:
```bash
snscrape --jsonl --progress twitter-user example_user > data.json
```

To use it within Python:
```python
import snscrape.modules.twitter as sntwitter
import pandas as pd

def scrape_tweets(username, limit=100):
    tweets = []
    for tweet in sntwitter.TwitterUserScraper(username).get_items():
        if len(tweets) >= limit:
            break
        tweets.append([tweet.date, tweet.content, tweet.likeCount, tweet.retweetCount])
    return pd.DataFrame(tweets, columns=['Date', 'Content', 'Likes', 'Retweets'])

# Example usage:
data = scrape_tweets("example_user", 50)
print(data.head())
```

### 🖥️ Running the System
1️⃣ Start the main application (`main.py`):
   ```bash
   python main.py
   ```
2️⃣ Test the `/detect-bot/` endpoint using Python:
   ```python
   import requests
   
   response = requests.post(
       "http://localhost:8000/detect-bot/",
       json={"profile_link": "https://x.com/example_user"}
   )
   print(response.json())
   ```

### 🌐 Streamlit Frontend
1️⃣ Run the Streamlit frontend:
   ```bash
   streamlit run frontend.py
   ```
2️⃣ Enter the profile link of the account you want to analyze and click **"Detect Bot."**

### 💻 CLI Frontend
1️⃣ Run the CLI script:
   ```bash
   python cli_frontend.py
   ```
2️⃣ Enter the profile link when prompted.

---

## ⚠️ Important Notes
📌 **snscrape Limitations:** Some users may have private or restricted accounts, which prevents scraping their data.
📌 **Rate Limits:** Although `snscrape` does not impose strict rate limits, excessive requests may trigger platform restrictions.
📌 **Data Accuracy:** The extracted data is dependent on the availability of public tweets and profile activity.

---

## 🚀 Deployment (Optional)
- 🌍 Deploy the system on a **cloud platform** (e.g., AWS, Azure, Heroku).
- 🎮 Host the **Streamlit frontend** locally or on **Streamlit Cloud**.

---

## 📚 Documentation
📖 For more details about the project, refer to the `Documentation.pdf` file. It includes:
✔️ **Architecture and implementation details**.
✔️ **Models and techniques used**.
✔️ **Setup instructions for local and cloud deployment**.

---

## 💚 License
This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## 📩 Contact
📧 For questions or feedback, contact: **codeitishant@gmail.com**

