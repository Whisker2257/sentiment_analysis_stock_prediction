#!/usr/bin/env python3
"""
Fixed Tweet Scraper for TwitterAPI.io (High-Capacity Staggered Version)

Scrapes tweets while ensuring an even distribution across the date range.

USAGE:
    python tweet_scraper.py "<query>" <start_date> <end_date> <max_tweets>

EXAMPLE:
    python tweet_scraper.py "PLTR OR $PLTR" 2024-10-01 2024-03-01 150000
"""

import os
import sys
import requests
import pandas as pd
from dotenv import load_dotenv
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()  # Load variables from .env file if present

def scrape_tweets(api_key: str, query: str, start_date: str, end_date: str, max_tweets=150000):
    """
    Scrape tweets with a per-day limit to distribute results evenly.

    :param api_key: TwitterAPI.io API key.
    :param query: The search query (e.g., "PLTR OR $PLTR").
    :param start_date: Start date (YYYY-MM-DD).
    :param end_date: End date (YYYY-MM-DD).
    :param max_tweets: Maximum number of tweets to retrieve in total.
    :return: List of tweet dicts.
    """
    base_url = "https://api.twitterapi.io/twitter/tweet/advanced_search"
    tweets_data = []
    total_tweets = 0
    daily_limit = min(1000, max_tweets)  # Max tweets per day

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    while start_dt <= end_dt and total_tweets < max_tweets:
        day_str = start_dt.strftime("%Y-%m-%d")
        next_day_str = (start_dt + timedelta(days=1)).strftime("%Y-%m-%d")  # Ensures full-day coverage
        logging.info(f"Fetching up to {daily_limit} tweets for {day_str}...")

        # Correct query format that worked in previous version
        query_with_time = f"({query}) since:{day_str} until:{next_day_str} -filter:replies lang:en"
        tweet_count = 0
        next_cursor = None

        while tweet_count < daily_limit and total_tweets < max_tweets:
            params = {
                "query": query_with_time,
                "queryType": "Latest",
                "max_results": min(100, daily_limit - tweet_count),
                "cursor": next_cursor if next_cursor else None
            }
            headers = {"X-API-Key": api_key}

            logging.info(f"Query: {query_with_time}")  # Debugging: Log actual query
            response = requests.get(base_url, headers=headers, params={k: v for k, v in params.items() if v is not None})

            if response.status_code != 200:
                logging.error(f"API request failed: {response.status_code} - {response.text}")
                break  # Stop querying for this day if an error occurs

            data = response.json()

            if 'tweets' not in data or not data['tweets']:
                logging.info(f"No tweets found for {day_str}. Moving to next day.")
                break  # Move to next day

            for tweet in data['tweets']:
                tweets_data.append({
                    "date": tweet.get('createdAt', 'UNKNOWN_DATE'),
                    "content": tweet.get('text', ''),
                    "retweetCount": tweet.get('retweetCount', 0),
                    "likeCount": tweet.get('likeCount', 0),
                    "replyCount": tweet.get('replyCount', 0),
                    "username": tweet.get('author', {}).get('userName', 'unknown')
                })
                tweet_count += 1
                total_tweets += 1

                if total_tweets >= max_tweets or tweet_count >= daily_limit:
                    break  # Stop if we hit max tweets

            next_cursor = data.get('next_cursor', None)
            if not next_cursor:
                break  # Stop paging if no more results

        start_dt += timedelta(days=1)  # Move to next day

    logging.info(f"Total tweets retrieved: {len(tweets_data)}")
    return tweets_data[:max_tweets]

def main():
    if len(sys.argv) < 5:
        print("Usage: python tweet_scraper.py <query> <start_date> <end_date> <max_tweets>")
        sys.exit(1)

    query = sys.argv[1]       
    start_date = sys.argv[2]  
    end_date = sys.argv[3]    
    max_tweets = int(sys.argv[4])
    
    # Get API key from environment variable
    API_KEY = os.getenv("TWITTERAPI_IO_KEY")
    if not API_KEY:
        logging.error("Error: TWITTERAPI_IO_KEY environment variable is not set.")
        sys.exit(1)

    logging.info(f"Scraping up to {max_tweets} tweets for: '{query}' from {start_date} to {end_date}")
    
    tweets = scrape_tweets(API_KEY, query, start_date, end_date, max_tweets)
    df = pd.DataFrame(tweets)
    
    # Save results to CSV
    output_filename = f"tweets_{query.replace(' ', '_')}_{start_date}_{end_date}.csv"
    output_path = os.path.join("data", "price_prediction_data", output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    logging.info(f"Saved {len(df)} tweets to {output_path}")

if __name__ == "__main__":
    main()
