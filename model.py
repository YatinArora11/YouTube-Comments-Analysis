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
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump


nltk.download('stopwords')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

negation_words = {
    'no', 'not', 'never', 'none', 'neither', 'nor',
    "don't", "doesn't", "didn't",
    "isn't", "aren't", "wasn't", "weren't",
    "hasn't", "haven't", "hadn't",
    "can't", "cannot", "couldn't",
    "won't", "wouldn't",
    "shouldn't", "mustn't"
}
final_stop_words = stop_words - negation_words

def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    
    tokens = [word for word in tokens if word not in final_stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

def load_data(file_path):
    col_names = ['ID', 'Category', 'Sentiment', 'Text']
    data = pd.read_csv(file_path, header=None, names=col_names)
    
    data['Text'] = data['Text'].fillna("")
    data['cleaned_text'] = data['Text'].apply(preprocess_text)
    return data

def train_model():
    data = load_data("twitter_training.csv")
    
    print("\nDataset label distribution:")
    print(data['Sentiment'].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        data['cleaned_text'],
        data['Sentiment'],
        test_size = 0.2, 
        random_state = 400
    )

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    svm_classifier = LinearSVC(class_weight = 'balanced')
    svm_classifier.fit(X_train_tfidf, y_train)

    y_pred = svm_classifier.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n--- Model Results ---")
    print(f"Model Accuracy: {accuracy: .4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    dump((svm_classifier, tfidf_vectorizer), 'sentiment_model.joblib')
    print("Model saved as 'sentiment_model.joblib'")

if __name__ == "__main__":
    train_model()
