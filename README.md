# 🐦 Twitter Sentiment Analysis (End-to-End Deep Learning NLP Project)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_APP_LINK_HERE)
[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-blue?logo=kaggle)](YOUR_KAGGLE_NOTEBOOK_LINK_HERE)

An end-to-end Natural Language Processing (NLP) project that classifies tweets into sentiment categories using a Bidirectional LSTM network enhanced with FastText pre-trained embeddings, deployed via an interactive Streamlit application.

---
## 🚀 Project Links

| Resource | Link |
|----------|------|
| 📓 Kaggle Notebook | [View Notebook](https://www.kaggle.com/code/samoura/twitter-sentiment-analysis) |
| 🌐 Live Demo | [Try the App](https://twitter-entity-sentiment-analysis-lstm.streamlit.app/) |
---
## 🎯 Project Overview
This project processes and classifies tweets associated with various entities (brands, games, platforms) into **4 main categories**:
* 🟢 **Positive:** Expresses satisfaction or admiration.
* 🔴 **Negative:** Clear complaints or frustration.
* 🔵 **Neutral:** General information or news.
* 🟠 **Irrelevant:** Tweets not directly related to the core context.

---

## 🛠️ Tech Stack & Skills
* **Languages:** Python
* **Libraries & Frameworks:** TensorFlow/Keras, Scikit-Learn, Pandas, NumPy, NLTK, Gensim (FastText), BeautifulSoup, Seaborn, Matplotlib.
* **Techniques:** Text Cleaning & Normalization, Tokenization & Padding, Handling Class Imbalance (Class Weights), TF-IDF, Pre-trained Word Embeddings, Bidirectional LSTM.
* **Deployment:** Streamlit

---

## 📁 Repository Structure
```text
├── app.py                  # Streamlit web application
├── requirements.txt        # Required Python packages
├── packages.txt            # System dependencies for deployment
├── best_model.keras        # Trained BiLSTM Model weights
└── tokenizer.pkl           # Saved Keras Tokenizer
