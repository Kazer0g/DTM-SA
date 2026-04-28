import os
import json
import csv
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
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC


RANDOM_STATE = 42
# Larger per-dataset sample improves generalization across domains.
SAMPLE_SIZE = 60000

ARTIFACTS_DIR = "."
MODEL_DIR = os.path.join(ARTIFACTS_DIR, "models")
METADATA_DIR = os.path.join(ARTIFACTS_DIR, "metadata")

MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

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


def build_candidate_models():
    """
    Candidate models keep the same core architecture (TF-IDF + LogisticRegression)
    and differ only in hyperparameters.
    """
    return [
        (
            "baseline",
            Pipeline([
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
        ),
        (
            "improved_tfidf_sublinear",
            Pipeline([
                ("tfidf", TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    max_features=80000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True
                )),
                ("classifier", LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    C=2.0,
                    random_state=RANDOM_STATE
                ))
            ])
        ),
        (
            "improved_regularized",
            Pipeline([
                ("tfidf", TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    max_features=70000,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True
                )),
                ("classifier", LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    C=1.2,
                    solver="lbfgs",
                    random_state=RANDOM_STATE
                ))
            ])
        ),
        (
            "linear_svm_word_ngrams",
            Pipeline([
                ("tfidf", TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    max_features=120000,
                    ngram_range=(1, 3),
                    min_df=2,
                    sublinear_tf=True
                )),
                ("classifier", LinearSVC(
                    C=1.2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                ))
            ])
        ),
        (
            "linear_svm_word_char",
            Pipeline([
                ("tfidf", FeatureUnion([
                    ("word_tfidf", TfidfVectorizer(
                        lowercase=True,
                        stop_words="english",
                        max_features=90000,
                        ngram_range=(1, 2),
                        min_df=2,
                        sublinear_tf=True
                    )),
                    ("char_tfidf", TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=2,
                        sublinear_tf=True
                    )),
                ])),
                ("classifier", LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                ))
            ])
        ),
        (
            "accuracy_focused_logreg",
            Pipeline([
                ("tfidf", TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    max_features=120000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.99,
                    sublinear_tf=True
                )),
                ("classifier", LogisticRegression(
                    max_iter=2000,
                    class_weight=None,
                    C=2.5,
                    solver="lbfgs",
                    random_state=RANDOM_STATE
                ))
            ])
        ),
        (
            "accuracy_focused_logreg_ngrams3",
            Pipeline([
                ("tfidf", TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    max_features=140000,
                    ngram_range=(1, 3),
                    min_df=2,
                    max_df=0.99,
                    sublinear_tf=True
                )),
                ("classifier", LogisticRegression(
                    max_iter=2500,
                    class_weight=None,
                    C=2.0,
                    solver="lbfgs",
                    random_state=RANDOM_STATE
                ))
            ])
        ),
        (
            "high_data_accuracy_logreg",
            Pipeline([
                ("tfidf", TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    max_features=220000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True
                )),
                ("classifier", LogisticRegression(
                    max_iter=3000,
                    class_weight=None,
                    C=1.0,
                    solver="lbfgs",
                    random_state=RANDOM_STATE
                ))
            ])
        ),
    ]


def select_best_model(X_train, y_train):
    """
    Select the best candidate by accuracy on a validation split.
    This improves model quality while staying deterministic and lightweight.
    """
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y_train
    )

    best_name = None
    best_model = None
    best_score = -1.0
    best_f1_for_tie_break = -1.0
    model_scores = []

    for model_name, candidate in build_candidate_models():
        print(f"Training candidate model: {model_name}")
        candidate.fit(X_subtrain, y_subtrain)
        val_pred = candidate.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_pred)
        val_f1_macro = f1_score(y_val, val_pred, average="macro", zero_division=0)
        model_scores.append({
            "model_name": model_name,
            "validation_accuracy": float(val_accuracy),
            "validation_f1_macro": float(val_f1_macro),
        })
        print(f"Validation accuracy ({model_name}): {val_accuracy:.4f}")
        print(f"Validation macro F1 ({model_name}): {val_f1_macro:.4f}")

        better_accuracy = val_accuracy > best_score
        tie_with_better_f1 = val_accuracy == best_score and val_f1_macro > best_f1_for_tie_break
        if better_accuracy or tie_with_better_f1:
            best_score = val_accuracy
            best_f1_for_tie_break = val_f1_macro
            best_name = model_name
            best_model = candidate

    print(f"Selected model: {best_name} (validation accuracy={best_score:.4f}, macro F1={best_f1_for_tie_break:.4f})")
    # Refit selected model on the full training split used for final evaluation.
    best_model.fit(X_train, y_train)
    return best_name, best_model, model_scores, best_score, best_f1_for_tie_break


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


def save_metadata(data, metrics, selected_model_name, candidate_scores):
    print("Saving model metadata...")
    vectorizer_step = metrics["model_snapshot"]["tfidf"]
    classifier_step = metrics["model_snapshot"]["classifier"]

    vectorizer_info = {
        "type": type(vectorizer_step).__name__,
    }
    if isinstance(vectorizer_step, TfidfVectorizer):
        vectorizer_info.update({
            "lowercase": bool(vectorizer_step.lowercase),
            "stop_words": vectorizer_step.stop_words,
            "max_features": vectorizer_step.max_features,
            "ngram_range": list(vectorizer_step.ngram_range),
            "min_df": vectorizer_step.min_df,
            "max_df": vectorizer_step.max_df,
            "sublinear_tf": bool(vectorizer_step.sublinear_tf)
        })
    elif isinstance(vectorizer_step, FeatureUnion):
        vectorizer_info["transformers"] = [name for name, _ in vectorizer_step.transformer_list]
        vectorizer_info["notes"] = "FeatureUnion of word and character TF-IDF branches"

    classifier_info = {
        "type": type(classifier_step).__name__,
    }
    if isinstance(classifier_step, LogisticRegression):
        classifier_info.update({
            "max_iter": classifier_step.max_iter,
            "class_weight": classifier_step.class_weight,
            "C": classifier_step.C,
            "solver": classifier_step.solver
        })
    elif isinstance(classifier_step, LinearSVC):
        classifier_info.update({
            "class_weight": classifier_step.class_weight,
            "C": classifier_step.C,
            "loss": classifier_step.loss
        })
    metadata = {
        "model_name": "sentiment_tfidf_logistic_regression",
        "model_type": "TF-IDF + Logistic Regression",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "selected_candidate": selected_model_name,
        "candidate_validation_scores": candidate_scores,

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
            "vectorizer": vectorizer_info,
            "classifier": classifier_info
        },

        "metrics": {
            key: value for key, value in metrics.items() if key != "model_snapshot"
        },

        "saved_files": {
            "vectorizer": VECTORIZER_PATH,
            "model": MODEL_PATH,
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
    data["group"] = data["category"]
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

    print("\n6. Training and selecting the best model...")
    selected_model_name, model, candidate_scores, best_val_acc, best_val_f1 = select_best_model(X_train, y_train)
    print("Best candidate selected and refit on full training split")

    print("\n7. Evaluating model on test set...")
    metrics = calculate_metrics(model, X_test, y_test)
    metrics["validation_accuracy_best_candidate"] = float(best_val_acc)
    metrics["validation_f1_macro_best_candidate"] = float(best_val_f1)
    metrics["model_snapshot"] = {
        "tfidf": model.named_steps["tfidf"],
        "classifier": model.named_steps["classifier"],
    }
    print("Evaluation completed")

    print("\n8. Saving model artifacts...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)

    # Save train/test datasets
    data_dir = "../data"
    os.makedirs(data_dir, exist_ok=True)
    
    train_data = pd.DataFrame({
        "text": X_train,
        "sentiment": y_train,
        "group": data.loc[X_train.index, "group"],
    })
    test_data = pd.DataFrame({
        "text": X_test,
        "sentiment": y_test,
        "group": data.loc[X_test.index, "group"],
    })
    
    train_data.to_csv(os.path.join(data_dir, "train.csv"), index=False, quoting=csv.QUOTE_ALL)
    test_data.to_csv(os.path.join(data_dir, "test.csv"), index=False, quoting=csv.QUOTE_ALL)
    
    print("Train data saved to:", os.path.join(data_dir, "train.csv"))
    print("Test data saved to:", os.path.join(data_dir, "test.csv"))

    # Save individual components for pipeline compatibility
    joblib.dump(model.named_steps['tfidf'], VECTORIZER_PATH)
    joblib.dump(model.named_steps['classifier'], MODEL_PATH)
    
    save_metadata(data, metrics, selected_model_name, candidate_scores)

    print("\nTraining completed successfully!")
    print("Vectorizer saved:", VECTORIZER_PATH)
    print("Model saved:", MODEL_PATH)
    print("Metadata saved:", METADATA_PATH)
    print("Accuracy:", round(metrics["accuracy"], 4))
    print("F1 macro:", round(metrics["f1_macro"], 4))
    print("F1 weighted:", round(metrics["f1_weighted"], 4))


if __name__ == "__main__":
    main()
