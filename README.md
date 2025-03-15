# 🕵️‍♂️ Bot Detector

## 📌 Overview
This project aims to develop a **machine learning-based system** for detecting bots on the social media platform **X (formerly Twitter)**. Bots are automated accounts that mimic human behavior and can be used for malicious purposes, such as spreading misinformation, spamming, or manipulating public opinion. By leveraging numerical features (e.g., retweet counts, mention counts) and advanced machine learning techniques, this system identifies bot accounts with high accuracy.

🚀 Developed by **Team Cyfer-Trace, SSPU**.

---

## ✨ Features
✔️ Detects bots based on **text patterns, posting behavior, and engagement metrics**.
✔️ Provides a **confidence score** indicating the likelihood of an account being a bot.
✔️ Generates structured outputs summarizing **bot classification results**.
✔️ Supports **real-time analysis** of social media profiles.

---

## 📜 Acknowledgements
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
- 🐍 Python **3.8+**
- 🔑 Access to the **X API (Bearer Token required)**
- 📦 Required libraries: `tweepy`, `pandas`, `numpy`, `scikit-learn`, `transformers`, `fasttext`, `joblib`

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
3️⃣ **Update API credentials:**
   Create a `config.json` file in the root directory with your X API credentials:
   ```json
   {
       "x": {
           "bearer_token": "your_bearer_token_here"
       }
   }
   ```
   **⚠️ Note:** Ensure the token has **read-only access** to public data. If you encounter a `401 Unauthorized` error, verify that the token is valid and correctly configured.

4️⃣ **Download pre-trained models and preprocessing objects:**
   Place the following files in the `models/` directory:
   - 📂 `distilbert_model/`
   - 📂 `fasttext_model.bin`
   - 📂 `lgbm_model.pkl`
   - 📂 `tfidf_vectorizer.pkl`
   - 📂 `scaler.pkl`

---

## 🚀 Usage

### 🖥️ Backend API
1️⃣ Start the backend server:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
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
📌 **API Rate Limits:** The X API imposes rate limits on the number of requests within a 15-minute window. If exceeded, you may encounter a `429 Too Many Requests` error. The system automatically retries after waiting for the rate limit reset.
📌 **Invalid API Key:** If you see a `401 Unauthorized` error, ensure your API key in `config.json` is correct and has the necessary permissions.
📌 **Waiting Time:** Due to rate limits, some requests may take longer to process. The system includes delays between requests to avoid hitting rate limits.

---

## 🚀 Deployment(Optional)
- 🌍 Deploy the backend API on a **cloud platform** (e.g., AWS, Azure, Heroku).
- 🎛️ Host the **Streamlit frontend** locally or on **Streamlit Cloud**.

---

## 📚 Documentation
📖 For more details about the project, refer to the `Documentation.pdf` file. It includes:
✔️ **Architecture and implementation details**.
✔️ **Models and techniques used**.
✔️ **Setup instructions for local and cloud deployment**.

---

## 📜 License
This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## 📩 Contact
📧 For questions or feedback, contact: **codeitishant@gmail.com**