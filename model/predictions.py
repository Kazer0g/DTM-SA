import os
from pathlib import Path
from collections import Counter

import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "sentiment_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"


def map_five_class_label(label):
    if label in [0, 1]:
        return "negative"
    if label == 2:
        return "neutral"
    return "positive"


def map_tweet_label(label):
    if label == 0:
        return "negative"
    if label == 1:
        return "neutral"
    return "positive"


def load_movies():
    dataset = load_dataset("SetFit/sst5", split="train")
    return pd.DataFrame([
        {
            "text": item["text"],
            "category": "movies",
            "source_dataset": "SetFit/sst5",
        }
        for item in dataset
    ])


def load_restaurants():
    dataset = load_dataset("Yelp/yelp_review_full", split="train")
    return pd.DataFrame([
        {
            "text": item["text"],
            "category": "restaurants",
            "source_dataset": "Yelp/yelp_review_full",
        }
        for item in dataset
    ])


def load_tweets():
    dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment", split="train")
    return pd.DataFrame([
        {
            "text": item["text"],
            "category": "tweets",
            "source_dataset": "cardiffnlp/tweet_eval",
        }
        for item in dataset
    ])


def build_model() -> Pipeline:
    vectorizer = joblib.load(VECTORIZER_PATH)
    classifier = joblib.load(MODEL_PATH)
    return Pipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier),
    ])


def predict_dataset(model: Pipeline, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    df = df.copy()
    df["model_input"] = df["category"] + " " + df["text"].astype(str)
    df["predicted_sentiment"] = model.predict(df["model_input"].tolist())
    return df


def print_summary(name: str, df: pd.DataFrame):
    total = len(df)
    counts = Counter(df["predicted_sentiment"])
    print(f"\n=== {name} Dataset ===")
    print(f"Total rows: {total}")
    for sentiment, count in counts.most_common():
        print(f"  {sentiment}: {count}")
    print("Sample predictions:")
    sample = df.head(10)
    for idx, row in sample.iterrows():
        print(f"  [{row['category']}] {row['text'][:120]!r} -> {row['predicted_sentiment']}")


def main():
    print("Loading trained model...")
    model = build_model()

    predicted_dfs = []
    datasets = [
        ("movies", load_movies()),
        ("restaurants", load_restaurants()),
        ("tweets", load_tweets()),
    ]

    for name, df in datasets:
        print(f"\nLoading {name} dataset ({len(df)} rows)...")
        predicted_df = predict_dataset(model, df, name)
        predicted_dfs.append(predicted_df)
        print_summary(name, predicted_df)

    combined = pd.concat(predicted_dfs, ignore_index=True)
    combined_counts = Counter(combined["predicted_sentiment"])
    print("\n=== Combined Dataset Summary ===")
    print(f"Total rows: {len(combined)}")
    for sentiment, count in combined_counts.most_common():
        print(f"  {sentiment}: {count}")


if __name__ == "__main__":
    main()
