# app.py

import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# Download resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# Rule-based fallback for very short inputs
positive_words = ['good', 'great', 'awesome', 'super', 'fantastic', 'love']
negative_words = ['bad', 'terrible', 'worst', 'hate', 'awful']

def rule_based_sentiment(text):
    text_lower = text.lower()
    for word in positive_words:
        if word in text_lower:
            return "Positive 😀"
    for word in negative_words:
        if word in text_lower:
            return "Negative 😡"
    return None

# Prediction function
def predict_sentiment(text):
    # Use rule-based for very short text (<3 words)
    if len(text.split()) <= 3:
        rule_result = rule_based_sentiment(text)
        if rule_result:
            return rule_result
    
    # Otherwise use ML model
    text_cleaned = clean_text(text)
    text_vectorized = tfidf.transform([text_cleaned])
    prediction = model.predict(text_vectorized)[0]
    return "Positive 😀" if prediction == 1 else "Negative 😡"

# Streamlit UI
st.title("🎭 Sentiment Analyzer")
st.write("Enter a review or text to analyze sentiment")

user_input = st.text_area("Enter text here:")

if st.button("Analyze"):
    if user_input.strip():
        result = predict_sentiment(user_input)
        st.success(f"Sentiment: {result}")
    else:
        st.warning("Please enter some text!")
