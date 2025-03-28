#!/usr/bin/env python3

"""
download_prepare_dataset.py

1) Loads the 'zeroshot/twitter-financial-news-sentiment' dataset from Hugging Face.
2) Maps the integer labels to {bearish, bullish, neutral}.
3) Renames the text column to 'tweet_text'.
4) Saves the final dataset as data/sentiment_training_data/fin_tweets.csv.
"""

import os
import pandas as pd
from datasets import load_dataset

def main():
    # 1) Load the dataset
    print("Loading dataset from Hugging Face: 'zeroshot/twitter-financial-news-sentiment' ...")
    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
    
    # Print a sample for debugging
    print("Sample example from the dataset:", dataset[0])
    # e.g. might show something like {'text': '...', 'label': 0}

    # 2) Convert to a pandas DataFrame
    df = pd.DataFrame(dataset)

    # 3) Inspect columns
    print("Columns in the dataset:", df.columns)
    # Expecting something like: ['text', 'label']

    # We'll define the columns we're expecting
    text_col = 'text'
    label_col = 'label'

    # Confirm these columns actually exist
    if text_col not in df.columns:
        raise ValueError(f"Expected '{text_col}' column not found in dataset columns: {df.columns}")
    if label_col not in df.columns:
        raise ValueError(f"Expected '{label_col}' column not found in dataset columns: {df.columns}")

    # 4) Map integer labels (0,1,2) to {bearish, bullish, neutral}
    def map_label(original_label):
        """
        Dataset docs say:
            0 -> Bearish
            1 -> Bullish
            2 -> Neutral
        We'll map them to string labels that our pipeline expects:
            0 => 'bearish'
            1 => 'bullish'
            2 => 'neutral'
        """
        if original_label == 0:
            return 'bearish'
        elif original_label == 1:
            return 'bullish'
        else:
            return 'neutral'
    
    # Apply the mapping
    df['mapped_label'] = df[label_col].apply(map_label)

    # 5) Rename columns to match the pipeline requirement: 'tweet_text' and 'label'
    df.rename(columns={text_col: 'tweet_text'}, inplace=True)
    # Move the mapped string label into the 'label' column
    df['label'] = df['mapped_label']

    # Keep only 'tweet_text' and 'label'
    final_df = df[['tweet_text', 'label']]

    # 6) Create the data/sentiment_training_data directory if it doesn't exist
    raw_dir = os.path.join("data", "sentiment_training_data")
    os.makedirs(raw_dir, exist_ok=True)

    # 7) Save final DataFrame as CSV
    output_path = os.path.join(raw_dir, "fin_tweets.csv")
    final_df.to_csv(output_path, index=False)
    print(f"Saved {len(final_df)} rows to {output_path}")
    print("Done.")

if __name__ == "__main__":
    main()
