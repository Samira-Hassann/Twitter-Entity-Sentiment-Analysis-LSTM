import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import html
from bs4 import BeautifulSoup
import emoji
import gdown
import os

# Set Page Config
st.set_page_config(
    page_title="Twitter Sentiment Analysis", page_icon="🐦", layout="centered"
)

# Google Drive File ID for the Model
MODEL_FILE_ID = "1Jw5GnCgf_x0xOYKimSDv_wwLuFs5R8i_"
MODEL_PATH = "best_model.keras"
TOKENIZER_PATH = "tokenizer (1).pkl"


@st.cache_resource
def load_assets():
  # Download model from Google Drive if not exists locally
  if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
    with st.spinner(
        "Downloading model from Google Drive... Please wait a moment ⏳"
    ):
      gdown.download(url, MODEL_PATH, quiet=False)

  # Load Model
  model = load_model(MODEL_PATH)

  # Load Tokenizer from local repo repository
  with open(TOKENIZER_PATH, "rb") as handle:
    tokenizer = pickle.load(handle)

  return model, tokenizer


model, tokenizer = load_assets()

MAX_LEN = 300


# Text Cleaning Function (matching your Kaggle notebook pipeline perfectly)
def clean_text(text):
  if not isinstance(text, str):
    return ""
  text = BeautifulSoup(text, "html.parser").get_text()
  text = html.unescape(text)
  text = text.lower()
  text = re.sub(r"[‘’´`′]", "'", text)
  text = re.sub(r"\b(\w+)\s+'\s+t\b", r"\1n't", text)
  text = re.sub(r"\b(\w+)\s+'\s+re\b", r"\1're", text)
  text = re.sub(r"\b(\w+)\s+'\s+ve\b", r"\1've", text)
  text = re.sub(r"\b(\w+)\s+'\s+ll\b", r"\1'll", text)
  text = re.sub(r"\b(\w+)\s+'\s+d\b", r"\1'd", text)
  text = re.sub(r"\b(\w+)\s+'\s+m\b", r"\1'm", text)
  text = re.sub(r"https?://\S+|www\.\S+", " <url> ", text)
  text = re.sub(r"pic\.twitter\.com/\S+", " <image> ", text)
  text = re.sub(r"@\w+", " <user> ", text)
  text = re.sub(r"#(\w+)", r"\1", text)
  text = re.sub(r"\brt\b", " ", text)
  text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
  text = emoji.demojize(text, delimiters=(" ", " "))
  text = re.sub(r"\s+", " ", text).strip()
  return text


# UI Layout
st.title("🐦 Twitter Sentiment Analysis App")
st.write(
    "Classify tweet sentiments into **Positive**, **Negative**, **Neutral**, or"
    " **Irrelevant** using a Deep Learning BiLSTM model."
)

user_input = st.text_area(
    "Enter a tweet to analyze:",
    placeholder="Type something like: I love this game, but the server lag is"
    " terrible...",
)

if st.button("Predict Sentiment", type="primary"):
  if user_input.strip() == "":
    st.warning("Please enter a valid text.")
  else:
    # Preprocess & Predict using exact notebook pipeline
    cleaned = clean_text(user_input)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    prediction = model.predict(padded)
    pred_class = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction))

    # Map classes based on LabelEncoder order:
    # ['Irrelevant', 'Negative', 'Neutral', 'Positive']
    classes_map = {
        0: ("Irrelevant", "🟠"),
        1: ("Negative", "🔴"),
        2: ("Neutral", "🔵"),
        3: ("Positive", "🟢"),
    }

    label_name, emoji_icon = classes_map[pred_class]

    st.markdown("---")
    st.subheader("Results:")
    st.markdown(
        f"### Prediction: {emoji_icon} **{label_name}** (Confidence:"
        f" {confidence:.2%})"
    )
