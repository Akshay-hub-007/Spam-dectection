# import pickle
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

# from sklearn.feature_extraction.text import TfidfVectorizer

# # Download necessary NLTK data (only once)
# nltk.download('stopwords')
# nltk.download('punkt')

# # Load the trained model and vectorizer
# with open("spam_model.pkl", "rb") as f:
#     model = pickle.load(f)

# with open("tfidf_vectorizer.pkl", "rb") as f:
#     tfidf = pickle.load(f)

# # Stopword set
# stop_words = set(stopwords.words('english'))

# # Function to clean text
# def remove_stopwords(text):
#     tokens = word_tokenize(text.lower())
#     filtered = [word for word in tokens if word.isalnum() and word not in stop_words]
#     return ' '.join(filtered)

# # Sample input
# text = "You have won a free vacation!"
# clean_text = remove_stopwords(text)

# # Transform using the loaded TF-IDF vectorizer (not a new one)
# vector = tfidf.transform([clean_text])

# # Predict
# prediction = model.predict(vector)

# # Output
# print("Spam" if prediction[0] == "spam" else "Ham")

# main.py
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download once
nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
# Load model and vectorizer
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

stop_words = set(stopwords.words('english'))

def clean_text(text):
    tokens = word_tokenize(text.lower())
    filtered = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered)

def predict_spam(text):
    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]
    return prediction
