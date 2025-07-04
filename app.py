# app.py
import streamlit as st
from main import predict_spam
import nltk
nltk.download('punkt')

st.set_page_config(page_title="Spam Detector", page_icon="📧")

st.title("📧 Spam Detector")
st.markdown("Enter a message and check if it's **Spam** or **Ham**.")

user_input = st.text_area("✏️ Your message:")

if st.button("🔍 Predict"):
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        result = predict_spam(user_input)
        if result == "spam" or result == 1:
            st.error("🚨 This message is **Spam**.")
        else:
            st.success("✅ This message is **Ham** (Not Spam).")
