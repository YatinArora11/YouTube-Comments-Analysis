import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

nltk.download('stopwords')
tokenizer = RegexpTokenizer(r'\w+')

def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Comment'] = data['Comment'].fillna("")
    data['cleaned_text'] = data['Comment'].apply(preprocess_text)
    return data

def train_model():
    data = load_data(r"C:\Users\Asus\OneDrive\Desktop\Projects\sentimental_analysis\YoutubeCommentsDataSet2.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        data['cleaned_text'],
        data['Sentiment'],
        test_size = 0.2, 
        random_state = 400
    )

    tfidf_vectorizer = TfidfVectorizer(max_features = 1000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    svm_classifier = LinearSVC()
    svm_classifier.fit(X_train_tfidf, y_train)

    y_pred = svm_classifier.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy: .4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    dump((svm_classifier, tfidf_vectorizer), 'sentiment_model.joblib')
    print("Model saved as 'sentiment_model.joblib'")

if __name__ == "__main__":
    train_model()
