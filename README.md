# Transformer-Based Tweet Sentiment Analysis & Random Forest Stock Prediction

This repository contains scripts for performing transformer-based Tweet Sentiment Analysis to power a Random Forest Stock Prediction model, illustrated using Palantir Technologies Inc. ($PLTR) as an example. All scripts and code are adequately commented for clarity.

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- A Twitter API key from [twitterapi.io](https://twitterapi.io) (Note: API usage incurs a cost of $0.15 per 1000 tweets.)

### Environment Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd <repository-folder>
```

2. **Set up your environment variables:**
Create a `.env` file at the root of your project containing your Twitter API key:
```env
TWITTERAPI_IO_KEY=your_twitter_api_key_here
```

3. **Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Running the Project

### Step 1: Download Sentiment Analysis Data

Run the following script to download and prepare the dataset:
```bash
python scripts/download_prepare_dataset.py
```

### Step 2: Train Sentiment Analysis Model

Train the transformer-based sentiment analysis model. (Estimated training time: ~3 hours on a 2020 Mac Pro)
```bash
python scripts/sentiment_finetuning.py
```

### Step 3: Scrape Tweets

Gather tweets for the stock of interest over at least a 5-month period:
```bash
python scripts/tweet_scraper.py 'PLTR OR $PLTR' 2024-10-01 2025-03-01 150000
```

(Note: Ensure your `.env` file is correctly set up with your Twitter API key.)

### Step 4: Train Random Forest Price Prediction Model

Use the sentiment analysis data to train the Random Forest model for price prediction:
```bash
python scripts/train_price_prediction.py
```

## Using the Model for Other Stocks

If the prediction accuracy significantly outperforms random chance (coin flip), you can use the trained pipeline for predicting other stocks:

1. Scrape tweets for the desired stock ticker:
```bash
python scripts/tweet_scraper.py 'your stock ticker here' yyyy-mm-dd yyyy-mm-dd number_of_tweets_here
```

2. Run your sentiment analysis and price prediction models on the gathered tweets to predict next-day stock behavior.

---

Nashe Gumbo
