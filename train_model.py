import os
import json
import joblib
import pandas as pd

from datetime import datetime
from datasets import load_dataset

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


RANDOM_STATE = 42
SAMPLE_SIZE = 12000

ARTIFACTS_DIR = "artifacts"
MODEL_DIR = os.path.join(ARTIFACTS_DIR, "models")
METADATA_DIR = os.path.join(ARTIFACTS_DIR, "metadata")

MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.pkl")
METADATA_PATH = os.path.join(METADATA_DIR, "sentiment_model_metadata.json")


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
            "label": map_five_class_label(item["label"]),
            "category": "movies",
            "source_dataset": "SetFit/sst5"
        }
        for item in dataset
    ])


def load_restaurants():
    dataset = load_dataset("Yelp/yelp_review_full", split="train")
    return pd.DataFrame([
        {
            "text": item["text"],
            "label": map_five_class_label(item["label"]),
            "category": "restaurants",
            "source_dataset": "Yelp/yelp_review_full"
        }
        for item in dataset
    ])


def load_tweets():
    dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment", split="train")
    return pd.DataFrame([
        {
            "text": item["text"],
            "label": map_tweet_label(item["label"]),
            "category": "tweets",
            "source_dataset": "cardiffnlp/tweet_eval"
        }
        for item in dataset
    ])


def sample_dataset(df):
    if len(df) <= SAMPLE_SIZE:
        return df

    return df.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )


def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=50000,
            ngram_range=(1, 2)
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])


def calculate_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)

    labels = ["negative", "neutral", "positive"]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0
        ),
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(
                y_test,
                y_pred,
                labels=labels
            ).tolist()
        }
    }


def save_metadata(data, metrics):
    metadata = {
        "model_name": "sentiment_tfidf_logistic_regression",
        "model_type": "TF-IDF + Logistic Regression",
        "created_at": datetime.now().isoformat(timespec="seconds"),

        "random_state": RANDOM_STATE,
        "sample_size_per_dataset": SAMPLE_SIZE,

        "training_data": {
            "total_rows": int(len(data)),
            "rows_by_category": data["category"].value_counts().to_dict(),
            "rows_by_label": data["label"].value_counts().to_dict(),
            "rows_by_source_dataset": data["source_dataset"].value_counts().to_dict()
        },

        "datasets": {
            "movies": "SetFit/sst5",
            "restaurants": "Yelp/yelp_review_full",
            "tweets": "cardiffnlp/tweet_eval"
        },

        "label_format": [
            "negative",
            "neutral",
            "positive"
        ],

        "model_parameters": {
            "vectorizer": {
                "type": "TfidfVectorizer",
                "lowercase": True,
                "stop_words": "english",
                "max_features": 50000,
                "ngram_range": [1, 2]
            },
            "classifier": {
                "type": "LogisticRegression",
                "max_iter": 1000,
                "class_weight": "balanced"
            }
        },

        "metrics": metrics,

        "saved_files": {
            "model": MODEL_PATH,
            "metadata": METADATA_PATH
        }
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)


def main():
    movies = sample_dataset(load_movies())
    restaurants = sample_dataset(load_restaurants())
    tweets = sample_dataset(load_tweets())

    data = pd.concat(
        [movies, restaurants, tweets],
        ignore_index=True
    )

    data = data.dropna(subset=["text", "label"])
    data = data[data["text"].str.strip() != ""]

    data["model_input"] = data["category"] + " " + data["text"]

    X_train, X_test, y_train, y_test = train_test_split(
        data["model_input"],
        data["label"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=data["label"]
    )

    model = build_model()
    model.fit(X_train, y_train)

    metrics = calculate_metrics(model, X_test, y_test)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    save_metadata(data, metrics)

    print("Model saved:", MODEL_PATH)
    print("Metadata saved:", METADATA_PATH)
    print("Accuracy:", round(metrics["accuracy"], 4))
    print("F1 macro:", round(metrics["f1_macro"], 4))
    print("F1 weighted:", round(metrics["f1_weighted"], 4))


if __name__ == "__main__":
    main()
