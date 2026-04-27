import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

TOKEN_PATTERN = re.compile(r"[A-Za-z']+")


def tokenize(text: str):
    return [token for token in TOKEN_PATTERN.findall(str(text).lower()) if len(token) > 1]


def validate_term_frequencies(df: pd.DataFrame, sample_size: int = 50) -> Dict[str, object]:
    sample = df.head(sample_size).copy()
    manual_counter = Counter()
    for text in sample["text"].fillna("").astype(str):
        manual_counter.update(tokenize(text))

    if sample.empty:
        return {"checked_rows": 0, "match_ratio": 1.0, "notes": "No rows to validate."}

    computed_counter = Counter()
    for text in sample["text"].fillna("").astype(str):
        computed_counter.update(tokenize(text))

    overlap = 0
    for term, count in manual_counter.items():
        overlap += min(count, computed_counter.get(term, 0))
    total = sum(manual_counter.values()) or 1
    return {
        "checked_rows": len(sample),
        "unique_terms": len(manual_counter),
        "token_total": total,
        "overlap_tokens": overlap,
        "match_ratio": overlap / total,
        "top_terms": manual_counter.most_common(20),
    }


def validate_sentiment(predictions_df: pd.DataFrame, labels_df: pd.DataFrame) -> Dict[str, object]:
    merged = predictions_df.merge(labels_df[["row_id", "label"]], on="row_id", how="inner")
    if merged.empty:
        return {"accuracy": None, "classification_report": {}, "confusion_matrix": []}

    y_true = merged["label"].astype(str)
    y_pred = merged["predicted_sentiment"].astype(str)
    return {
        "samples_compared": len(merged),
        "accuracy": accuracy_score(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Validate pipeline outputs.")
    parser.add_argument("--input-csv", required=True, help="Source CSV with label.")
    parser.add_argument(
        "--predictions-csv", required=True, help="Predictions from pipeline output."
    )
    parser.add_argument(
        "--output-json",
        default="outputs/validation_results.json",
        help="Where to save validation report.",
    )
    parser.add_argument("--sample-size", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    input_df = pd.read_csv(args.input_csv).reset_index().rename(columns={"index": "row_id"})
    input_df["row_id"] = input_df["row_id"].astype(str)
    predictions_df = pd.read_csv(args.predictions_csv)
    predictions_df["row_id"] = predictions_df["row_id"].astype(str)

    results = {
        "term_frequency_validation": validate_term_frequencies(
            input_df.rename(columns={input_df.columns[input_df.columns.get_loc("text")]: "text"}),
            sample_size=args.sample_size,
        ),
        "sentiment_validation": validate_sentiment(predictions_df, input_df),
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
