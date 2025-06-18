import streamlit as st
import joblib
import re

# Load your saved model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Text cleaning function
def clean_review(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    return text

# Streamlit UI
st.title("📦 Amazon Review Sentiment Analyzer")

user_input = st.text_area("Enter your product review:")

if st.button("Predict Sentiment"):
    cleaned = clean_review(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)

    sentiment = "👍 Positive" if prediction[0] == 1 else "👎 Negative"
    st.success(f"Prediction: {sentiment}")
