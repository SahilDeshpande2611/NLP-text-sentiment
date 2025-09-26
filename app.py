import streamlit as st
import joblib

st.title("Sentiment Analysis Sahil 🤖")
st.markdown("Type a sentence and I will predict the emotion!")

emotion_numbers = joblib.load("emotion_numbers.pkl")
st.write("Emotion label mapping:", emotion_numbers)

# Load model and objects once
model = joblib.load("regression_sentimentanalysis.pkl")
tfidf_vectorizer = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# Input text
user_text = st.text_area("Enter your sentence here:")

if st.button("Predict"):
    if user_text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Transform input
        input_vec = tfidf_vectorizer.transform([user_text])

        # Predict
        pred = model.predict(input_vec)[0]
        emotion = le.inverse_transform([pred])[0]

        # Show result
        st.success(f"✅ Predicted Sentiment: **{emotion}**")