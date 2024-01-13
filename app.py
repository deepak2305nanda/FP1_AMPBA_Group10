import numpy as np
import pandas as pd
import pickle
import streamlit as st
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import yfinance as yf

# Loading the trained machine learning model
pickle_in = open("model.sav", "rb")
model = pickle.load(pickle_in)

# Mapping dictionary for sentiment labels
sentiment_mapping = {0: 'Bearish', 1: 'Bullish'}

# Text preprocessing function
def preprocess_text(tweet):
    # Convert to lowercase
    tweet = tweet.lower()
    # Remove punctuation
    tweet = re.sub(r'[^\w\s]', '', tweet)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tweet)
    tweet = ' '.join([word for word in word_tokens if word not in stop_words])
    return tweet

# Predicting sentiment function
def predict_stock_sentiment(tweet):
    tweet = preprocess_text(tweet)
    # Predict sentiment using the loaded model
    prediction = model.predict([tweet])
    return prediction[0]

# Mapping sentiment to labels function
def map_sentiment(sentiment):
    return sentiment_mapping.get(sentiment, 'Unknown')

# Function to fetch stock data using yfinance
def get_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

# Main Streamlit app function
def main():
    st.title("ICICI Stock Sentiment Predictor - Group-10 AMPBA Co'24 Summer")
    
    # HTML styling for the app
    html_temp = """
    <div style="background-color: tomato; padding: 10px;">
    <h2 style="color: white; text-align: center;">Streamlit Stock Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # User input for the tweet
    Stock_Reviews = st.text_input("Tweet", "Type here")
    result = ""

    # Predict button click event
    if st.button("Predict"):
        # Predict sentiment and map to label
        sentiment = predict_stock_sentiment(Stock_Reviews)
        result = map_sentiment(sentiment)

    # Displaying the predicted sentiment
    st.success('The sentiment is {}'.format(result))

    # Fetching stock data and displaying the chart
    st.subheader("ICICI Stock Chart (Last 10 Years)")
    icici_stock_data = get_stock_data("ICICIBANK.BO", start_date="2012-01-01", end_date="2022-01-01")
    st.line_chart(icici_stock_data['Close'])

# Running the app
if __name__ == '__main__':
    main()
