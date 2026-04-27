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
from sklearn.preprocessing import LabelEncoder


RANDOM_STATE = 42
SAMPLE_SIZE = 12000

ARTIFACTS_DIR = "./model"
MODEL_DIR = os.path.join(ARTIFACTS_DIR, "models")
METADATA_DIR = os.path.join(ARTIFACTS_DIR, "metadata")

MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
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
    print("Loading movies dataset from SetFit/sst5...")
    dataset = load_dataset("SetFit/sst5", split="train")
    df = pd.DataFrame([
        {
            "text": item["text"],
            "label": map_five_class_label(item["label"]),
            "category": "movies",
            "source_dataset": "SetFit/sst5"
        }
        for item in dataset
    ])
    print(f"Loaded {len(df)} movie reviews")
    return df


def load_restaurants():
    print("Loading restaurants dataset from Yelp/yelp_review_full...")
    dataset = load_dataset("Yelp/yelp_review_full", split="train")
    df = pd.DataFrame([
        {
            "text": item["text"],
            "label": map_five_class_label(item["label"]),
            "category": "restaurants",
            "source_dataset": "Yelp/yelp_review_full"
        }
        for item in dataset
    ])
    print(f"Loaded {len(df)} restaurant reviews")
    return df


def load_tweets():
    print("Loading tweets dataset from cardiffnlp/tweet_eval...")
    dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment", split="train")
    df = pd.DataFrame([
        {
            "text": item["text"],
            "label": map_tweet_label(item["label"]),
            "category": "tweets",
            "source_dataset": "cardiffnlp/tweet_eval"
        }
        for item in dataset
    ])
    print(f"Loaded {len(df)} tweets")
    return df


def sample_dataset(df):
    if len(df) <= SAMPLE_SIZE:
        print(f"Dataset has {len(df)} samples, no sampling needed")
        return df

    print(f"Sampling {SAMPLE_SIZE} examples from {len(df)} total")
    return df.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )


def build_model():
    print("Building TF-IDF + Logistic Regression pipeline...")
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
    print("Saving model metadata...")
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
            "vectorizer": VECTORIZER_PATH,
            "model": MODEL_PATH,
            "label_encoder": LABEL_ENCODER_PATH,
            "metadata": METADATA_PATH
        }
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)


def main():
    print("Starting model training process...")
    
    print("\n1. Loading and sampling datasets...")
    movies = sample_dataset(load_movies())
    restaurants = sample_dataset(load_restaurants())
    tweets = sample_dataset(load_tweets())

    print("\n2. Combining datasets...")
    data = pd.concat(
        [movies, restaurants, tweets],
        ignore_index=True
    )
    print(f"Combined dataset has {len(data)} total samples")
    print("Label distribution:", data["label"].value_counts().to_dict())
    print("Category distribution:", data["category"].value_counts().to_dict())

    print("\n3. Preprocessing data...")
    initial_count = len(data)
    data = data.dropna(subset=["text", "label"])
    data = data[data["text"].str.strip() != ""]
    print(f"After preprocessing: {len(data)} samples (removed {initial_count - len(data)})")

    print("\n4. Creating model inputs...")
    data["model_input"] = data["category"] + " " + data["text"]
    print("Added category prefixes to texts")

    print("\n5. Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        data["model_input"],
        data["label"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=data["label"]
    )
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    print("\n6. Training the model...")
    model = build_model()
    print("Fitting model on training data...")
    model.fit(X_train, y_train)
    print("Model training completed")

    print("\n7. Creating label encoder...")
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    print("Label encoder created")

    print("\n8. Evaluating model on test set...")
    metrics = calculate_metrics(model, X_test, y_test)
    print("Evaluation completed")

    print("\n9. Saving model artifacts...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)

    # Save individual components for pipeline compatibility
    joblib.dump(model.named_steps['tfidf'], VECTORIZER_PATH)
    joblib.dump(model.named_steps['classifier'], MODEL_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    
    save_metadata(data, metrics)

    print("\nTraining completed successfully!")
    print("Vectorizer saved:", VECTORIZER_PATH)
    print("Model saved:", MODEL_PATH)
    print("Label encoder saved:", LABEL_ENCODER_PATH)
    print("Metadata saved:", METADATA_PATH)
    print("Accuracy:", round(metrics["accuracy"], 4))
    print("F1 macro:", round(metrics["f1_macro"], 4))
    print("F1 weighted:", round(metrics["f1_weighted"], 4))


if __name__ == "__main__":
    main()
