#!/usr/bin/env python3

"""
train_price_prediction.py

1) Loads tweets from CSV (scraped via snscrape for historical data).
2) Runs them through the fine-tuned FinancialBERT to get sentiment scores.
3) Aggregates daily sentiment, merges with stock price data.
4) Trains a RandomForest to predict daily price movement using a 14-day lookback window.

This version caches sentiment inference results (pickle) and downloaded stock data (CSV)
in data/price_prediction_data to speed up iterative development.
"""

import os
import re
import pandas as pd
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import yfinance as yf
import joblib

def custom_preprocess(text: str) -> str:
    # Replace URLs, remove hashtags, and clean repeated punctuation.
    text = re.sub(r'http\S+|www.\S+', '[URL]', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'([!?.])\1+', r'\1', text)
    return text

def load_finetuned_model(model_dir="./models/finetuned_finbert"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

def predict_sentiment(text, tokenizer, model, device):
    processed_text = custom_preprocess(text)
    inputs = tokenizer(processed_text, return_tensors='pt', truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    preds = torch.argmax(logits, dim=1).item()
    return preds  # 0: bearish, 1: neutral, 2: bullish

def main():
    # -------------------
    # 1) Load Tweets
    # -------------------
    tweets_csv = os.path.join("data", "price_prediction_data", "tweets_PLTR_OR_$PLTR_2024-10-01_2025-03-01.csv")
    if not os.path.exists(tweets_csv):
        raise FileNotFoundError(f"CSV file not found: {tweets_csv}")
    df_tweets = pd.read_csv(tweets_csv)
    df_tweets['date'] = pd.to_datetime(df_tweets['date'])

    # -------------------
    # 2) Load Finetuned Model
    # -------------------
    tokenizer, model, device = load_finetuned_model(model_dir=os.path.join("models", "finetuned_finbert"))

    # -------------------
    # 3) Run Sentiment Inference on Tweets (with caching)
    # -------------------
    sentiment_cache_path = os.path.join("data", "price_prediction_data", "tweets_sentiment.pkl")
    if os.path.exists(sentiment_cache_path):
        print("Loading cached tweet sentiments...")
        df_tweets = pd.read_pickle(sentiment_cache_path)
    else:
        print("Running sentiment inference on tweets...")
        df_tweets['sentiment_label'] = df_tweets['content'].apply(
            lambda x: predict_sentiment(x, tokenizer, model, device)
        )
        # Map labels: 0 -> -1 (bearish), 1 -> 0 (neutral), 2 -> +1 (bullish)
        label_map = {0: -1, 1: 0, 2: 1}
        df_tweets['sentiment_score'] = df_tweets['sentiment_label'].map(label_map)
        # Normalize retweet counts and compute weighted sentiment
        max_retweets = df_tweets['retweetCount'].max() if df_tweets['retweetCount'].max() > 0 else 1
        df_tweets['retweet_weight'] = df_tweets['retweetCount'] / max_retweets
        df_tweets['weighted_sentiment'] = df_tweets['retweet_weight'] * df_tweets['sentiment_score']
        # Cache the results
        df_tweets.to_pickle(sentiment_cache_path)
    
    # -------------------
    # 4) Aggregate Weighted Sentiment by Day
    # -------------------
    df_tweets['day'] = df_tweets['date'].dt.date
    df_daily_sentiment = df_tweets.groupby('day').agg({
        'weighted_sentiment': 'sum'
    }).reset_index()
    df_daily_sentiment.rename(columns={'weighted_sentiment': 'daily_weighted_sentiment'}, inplace=True)

    # -------------------
    # 5) Get Stock Price Data (with caching)
    # -------------------
    ticker_symbol = "PLTR"
    start_date = "2024-10-01"
    end_date   = "2025-03-01"
    stock_cache_path = os.path.join("data", "price_prediction_data", f"stock_data_{ticker_symbol}_{start_date}_{end_date}.csv")
    
    if os.path.exists(stock_cache_path):
        print("Loading cached stock data...")
        df_prices = pd.read_csv(stock_cache_path)
        df_prices['Date'] = pd.to_datetime(df_prices['Date'])
    else:
        print(f"Downloading stock data for {ticker_symbol} from {start_date} to {end_date}...")
        df_prices = yf.download(ticker_symbol, start=start_date, end=end_date)
        if df_prices.empty:
            raise ValueError("No stock price data downloaded. Check the ticker symbol and date range.")
        df_prices.reset_index(inplace=True)
        # Save the downloaded stock data to CSV
        df_prices.to_csv(stock_cache_path, index=False)
        print(f"Stock data saved to {stock_cache_path}")
    
    # Ensure proper formatting
    if isinstance(df_prices.columns, pd.MultiIndex):
        df_prices.columns = df_prices.columns.get_level_values(0)
    df_prices.rename(columns={"Date": "day"}, inplace=True)
    df_prices['day'] = df_prices['day'].dt.date

    # Merge stock price data with daily sentiment data
    df_merged = pd.merge(df_prices, df_daily_sentiment, on='day', how='left')
    df_merged['daily_weighted_sentiment'] = df_merged['daily_weighted_sentiment'].fillna(0.0)

    # -------------------
    # 6) Create Features & Labels using a 14-Day Lookback Window
    # -------------------
    df_merged.sort_values('day', inplace=True)
    # Use "Close" as the available price column.
    df_merged['price_diff'] = df_merged['Close'].diff()
    df_merged['price_movement'] = (df_merged['price_diff'] > 0).astype(int)
    # For day t, aggregate daily weighted sentiment from t-14 to t-1.
    df_merged['sentiment_14'] = df_merged['daily_weighted_sentiment'].rolling(window=14).sum().shift(1)
    # Use the previous day's closing price as a feature.
    df_merged['price_prev'] = df_merged['Close'].shift(1)
    # Drop rows with missing values from rolling and shift operations.
    df_merged.dropna(subset=['sentiment_14', 'price_prev', 'price_movement'], inplace=True)

    features = ['sentiment_14', 'price_prev']
    X = df_merged[features]
    y = df_merged['price_movement']

    # -------------------
    # 7) Train Random Forest Model
    # -------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)

    # -------------------
    # 8) Evaluate the Model
    # -------------------
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("RandomForest Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save the trained RandomForest model
    model_path = os.path.join("models", "randomforest_model.pkl")
    joblib.dump(rf, model_path)
    print(f"Saved RandomForest model to {model_path}")

if __name__ == "__main__":
    main()
