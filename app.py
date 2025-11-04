import streamlit as st
from joblib import load
import re
import string 
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from googleapiclient.discovery import build
import pandas as pd

nltk.download('stopwords')
tokenizer = RegexpTokenizer(r'\w+')
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

api_key = 'AIzaSyCCk4nvnt6E-MZgKr6sH0zVD29_KnrDVgs'

try:
    svm_classifier, tfidf_vectorizer = load('sentiment_model.joblib')
except FileNotFoundError:
    st.error("Model file not found. PLease train the model first.")
    st.stop()

def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    tokens = [word for word in tokens if word not in final_stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def analyze_sentiment(text):
    cleaned_text = preprocess_text(text)
    text_transformed = tfidf_vectorizer.transform([cleaned_text])
    return svm_classifier.predict(text_transformed)[0]

def get_youtube_comments(video_id, max_results = 100) :
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    next_page_token =None

    try:
        while len(comments) < max_results:
            request = youtube.commentThreads().list(
                part = "snippet",
                videoId = video_id,
                maxResults = min(100, max_results - len(comments)), 
                textFormat = "plainText",
                pageToken = next_page_token
            )
            response = request.execute()

            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break

    except Exception as e:
        st.error(f"Error fetching comments: {str(e)}")
        return []
    
    return comments [:max_results]
    
def extract_video_id(url):
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:\?|&|$)"
        ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

#UI
st.title("Youtube Comment Sentiment Analyzer")
st.markdown("Analyze sentiment of Youtube comments using SVM classifier")

url = st.text_input("Enter Youtube Video URL:")
analyze_button = st.button("Analyze Comments")

if analyze_button and url:
    video_id = extract_video_id(url)

    if not video_id:
        st.error("Invalid YouTube URL. Please enter a a valid URL.")
        st.stop()

    with st.spinner("Fetching Comments..."):
        comments = get_youtube_comments(video_id)
        
    st.write(f"Video ID: {video_id}, Fetched {len(comments)} comments")

    if not comments:
        st.warning("NO comments found for this video.")
        st.stop()
    results = []
    with st.spinner("Analyzing sentiments..."):
        for comment in comments:
            sentiment = analyze_sentiment(comment)
            results.append({
                "Comment" : comment,
                "Sentiment" : sentiment
            })

    df = pd.DataFrame(results)
    st.dataframe(df)

    st.subheader("Sentiment Distribution")
    sentiment_counts = df["Sentiment"].value_counts()
    st.bar_chart(sentiment_counts)

    if sentiment_counts.empty:
        st.warning("No sentiment results to display")
    else:
        positive_count = sentiment_counts.get('Positive', 0)
        negative_count = sentiment_counts.get('Negative', 0)
        neutral_count = sentiment_counts.get('Neutral', 0)
        irrelevant_count = sentiment_counts.get('Irrelevant', 0)

        # Find the dominant sentiment
        dominant_sentiment = sentiment_counts.idxmax()

        st.subheader("Overall Sentiment")
        
        # This logic should now work as expected
        if dominant_sentiment == 'Positive':
            st.success(f"Overall Positive Sentiment ({positive_count}/{len(df)} comments)")
        elif dominant_sentiment == 'Negative':
            st.error(f"Overall Negative Sentiment ({negative_count}/{len(df)} comments)")
        elif dominant_sentiment == 'Neutral':
            st.info(f"Overall Neutral Sentiment ({neutral_count}/{len(df)} comments)")
        elif dominant_sentiment == 'Irrelevant':
            st.info(f"Overall Irrelevant Sentiment ({irrelevant_count}/{len(df)} comments)")

        # --- CORRECTED CODE: Display all counts in columns ---
        st.subheader("All Sentiment Counts")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # FIX: Corrected variable name from positive_tcount to positive_count
            st.metric(label="Positive", value=positive_count)
        with col2:
            st.metric(label="Negative", value=negative_count)
        with col3:
            st.metric(label="Neutral", value=neutral_count)
        with col4:
            st.metric(label="Irrelevant", value=irrelevant_count)
