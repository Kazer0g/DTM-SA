#preprocess.py
#loads all 3 datasets, cleans text, balances classes, saves train/test splits
#usage: python preprocess.py [--max-per-class 12000] [--test-size 0.2] [--seed 42]

import argparse
import os
import re
import string

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

#keeping this inline to avoid nltk dependency
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "it", "its", "this",
    "that", "these", "those", "i", "you", "he", "she", "we", "they", "me",
    "him", "her", "us", "them", "my", "your", "his", "our", "their",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "as", "if", "then", "than", "so", "yet", "both", "either", "neither",
    "not", "no", "nor", "just", "also", "very", "too", "quite", "such",
    "up", "out", "about", "into", "through", "during", "before", "after",
    "over", "under", "again", "further", "once", "there", "here",
}


def clean_text(text: str, group: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text) #remove urls
    text = re.sub(r"<[^>]+>", " ", text) #remove html tags

    if group == "tweets":
        text = re.sub(r"@\w+", " ", text) #remove @mentions
        text = re.sub(r"#(\w+)", r" \1 ", text) #strip # but keep the word

    #remove punctuation except apostrophes (don't, wasn't etc.)
    text = text.translate(
        str.maketrans(
            string.punctuation.replace("'", ""),
            " " * (len(string.punctuation) - 1),
        )
    )

    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stop_words(text: str) -> str:
    tokens = text.split()
    return " ".join(t for t in tokens if t not in STOP_WORDS)


#sst5 has 5 classes, collapsing to 3
def map_sst5_label(label: int) -> str:
    if label <= 1:
        return "negative"
    if label == 2:
        return "neutral"
    return "positive"


def map_yelp_label(label: int) -> str:
    if label <= 1:
        return "negative"
    if label == 2:
        return "neutral"
    return "positive"


def map_tweeteval_label(label: int) -> str:
    mapping = {0: "negative", 1: "neutral", 2: "positive"}
    return mapping.get(label, "neutral")


def load_sst5(max_rows: int) -> pd.DataFrame:
    print("  loading sst5 (movies)...")
    ds = load_dataset("SetFit/sst5", split="train+validation+test")
    df = pd.DataFrame({"text": ds["text"], "raw_label": ds["label"]})
    df["label"] = df["raw_label"].apply(map_sst5_label)
    df["group"] = "movies"
    return df[["text", "label", "group"]].head(max_rows)


def load_yelp(max_rows: int) -> pd.DataFrame:
    print("  loading yelp (restaurants)...")
    ds = load_dataset("Yelp/yelp_review_full", split="train+test")
    df = pd.DataFrame({"text": ds["text"], "raw_label": ds["label"]})
    df["label"] = df["raw_label"].apply(map_yelp_label)
    df["group"] = "restaurants"
    return df[["text", "label", "group"]].head(max_rows)


def load_tweeteval(max_rows: int) -> pd.DataFrame:
    print("  loading tweeteval (tweets)...")
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment",
                      split="train+validation+test")
    df = pd.DataFrame({"text": ds["text"], "raw_label": ds["label"]})
    df["label"] = df["raw_label"].apply(map_tweeteval_label)
    df["group"] = "tweets"
    return df[["text", "label", "group"]].head(max_rows)


def balance(df: pd.DataFrame, max_per_class: int, seed: int) -> pd.DataFrame:
    #cap each class so the model doesn't overfit to yelp
    parts = []
    for label, group in df.groupby("label"):
        parts.append(group.sample(min(len(group), max_per_class), random_state=seed))
    return pd.concat(parts, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-class", type=int, default=12_000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--remove-stopwords", action="store_true")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)

    print("\nloading datasets...")
    dfs = []
    for loader in [load_sst5, load_yelp, load_tweeteval]:
        try:
            dfs.append(loader(max_rows=args.max_per_class * 4))
        except Exception as e:
            print(f"  warning: could not load dataset - {e}")

    if not dfs:
        raise RuntimeError("no datasets loaded, check internet connection")

    corpus = pd.concat(dfs, ignore_index=True)
    print(f"  raw total: {len(corpus):,} rows")

    print("\ncleaning text...")
    corpus["text"] = corpus.apply(
        lambda row: clean_text(row["text"], row["group"]), axis=1
    )
    if args.remove_stopwords:
        corpus["text"] = corpus["text"].apply(remove_stop_words)

    corpus = corpus[corpus["text"].str.strip() != ""].reset_index(drop=True)

    print("\nbalancing classes...")
    corpus = balance(corpus, args.max_per_class, args.seed)
    corpus = corpus.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    print(f"  total: {len(corpus):,} rows")
    print(f"  labels: {dict(corpus['label'].value_counts())}")
    print(f"  groups: {dict(corpus['group'].value_counts())}")

    print("\nsplitting and saving...")
    train_df, test_df = train_test_split(
        corpus,
        test_size=args.test_size,
        stratify=corpus["label"],
        random_state=args.seed,
    )

    corpus.to_csv("data/corpus.csv", index=False)
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print(f"  corpus.csv  -> {len(corpus):,} rows")
    print(f"  train.csv   -> {len(train_df):,} rows")
    print(f"  test.csv    -> {len(test_df):,} rows")


if __name__ == "__main__":
    main()
