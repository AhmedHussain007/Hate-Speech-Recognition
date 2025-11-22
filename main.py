import streamlit as st
import joblib

# Load trained TF-IDF + Logistic Regression model and vectorizer
model = joblib.load("best_model.pkl")      # Make sure this is the Logistic Regression model
vectorizer = joblib.load("vectorizer.pkl")

# Map numeric labels to human-readable text
label_map = {0: "Not Hate Text", 1: "Hate Text"}

def predict_text(text):
    """
    Predict if input text is hate speech or not using Logistic Regression
    """
    X_input = vectorizer.transform([text])
    pred_label = model.predict(X_input)[0]
    return label_map[pred_label]

# Streamlit UI
st.title("Hate Speech Detection")
st.write("Enter text below and the model will classify it as Hate Text or Not Hate Text using TF-IDF + Logistic Regression.")

# Input text area
user_input = st.text_area("Enter text here:")

# Prediction button
if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text to classify!")
    else:
        result = predict_text(user_input)
        st.success(f"Prediction: {result}")
