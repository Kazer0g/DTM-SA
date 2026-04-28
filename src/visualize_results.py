import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create one combined PNG with project charts."
    )
    parser.add_argument(
        "--scalability-csv",
        default="outputs/scalability_results.csv",
        help="Path to scalability CSV file.",
    )
    parser.add_argument(
        "--validation-json",
        default="outputs/validation_results.json",
        help="Path to validation JSON file.",
    )
    parser.add_argument(
        "--group-report-json",
        default="outputs/group_term_sentiment_report.json",
        help="Path to group term report JSON file.",
    )
    parser.add_argument(
        "--predictions-csv",
        default="outputs/predictions.csv",
        help="Path to predictions CSV file (optional but recommended).",
    )
    parser.add_argument(
        "--output-png",
        default="outputs/dashboard.png",
        help="Path to output combined PNG.",
    )
    parser.add_argument(
        "--top-n-terms",
        type=int,
        default=10,
        help="How many top terms to show in term charts.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_top_terms(
    group_report: Dict[str, Dict[str, List[Dict[str, int]]]], sentiment_key: str, top_n: int
) -> List[Tuple[str, int]]:
    aggregate: Dict[str, int] = {}
    if isinstance(group_report.get(sentiment_key), list):
        # New format: report already stores overall top terms by sentiment.
        for item in group_report.get(sentiment_key, []):
            term = str(item["term"])
            count = int(item["count"])
            aggregate[term] = aggregate.get(term, 0) + count
    else:
        # Backward compatibility: old format nested by group.
        for _group, payload in group_report.items():
            if isinstance(payload, dict):
                for item in payload.get(sentiment_key, []):
                    term = str(item["term"])
                    count = int(item["count"])
                    aggregate[term] = aggregate.get(term, 0) + count
    ranked = sorted(aggregate.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def main():
    args = parse_args()
    sns.set_theme(style="whitegrid")

    scalability_path = Path(args.scalability_csv)
    validation_path = Path(args.validation_json)
    group_report_path = Path(args.group_report_json)
    predictions_path = Path(args.predictions_csv)
    output_path = Path(args.output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scalability_df = pd.read_csv(scalability_path)
    validation = load_json(validation_path)
    group_report = load_json(group_report_path)

    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    fig.suptitle("Distributed Text Mining & Sentiment Dashboard", fontsize=18, y=0.98)

    # Chart 1: Runtime vs chunk size
    ax1 = axes[0, 0]
    runtime_df = scalability_df.sort_values(["max_docs", "chunk_size"])
    for doc_count, subset in runtime_df.groupby("max_docs"):
        ax1.plot(
            subset["chunk_size"],
            subset["elapsed_seconds"],
            marker="o",
            label=f"docs={doc_count}",
        )
    ax1.set_title("Runtime vs Chunk Size")
    ax1.set_xlabel("Chunk Size")
    ax1.set_ylabel("Elapsed Seconds")
    ax1.legend()

    # Chart 2: Speedup vs chunk size
    ax2 = axes[0, 1]
    speed_df = scalability_df.sort_values(["max_docs", "chunk_size"])
    for doc_count, subset in speed_df.groupby("max_docs"):
        ax2.plot(
            subset["chunk_size"],
            subset["speedup_vs_first_chunk"],
            marker="o",
            label=f"docs={doc_count}",
        )
    ax2.set_title("Speedup vs Chunk Size")
    ax2.set_xlabel("Chunk Size")
    ax2.set_ylabel("Speedup")
    ax2.legend()

    # Chart 3: Confusion matrix
    ax3 = axes[0, 2]
    matrix = validation.get("sentiment_validation", {}).get("confusion_matrix", [])
    report_keys = list(
        validation.get("sentiment_validation", {})
        .get("classification_report", {})
        .keys()
    )
    labels = [k for k in report_keys if k not in ("accuracy", "macro avg", "weighted avg")]
    if matrix and labels:
        matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
        sns.heatmap(matrix_df, annot=True, fmt="d", cmap="Blues", ax=ax3)
        ax3.set_title("Confusion Matrix")
        ax3.set_xlabel("Predicted")
        ax3.set_ylabel("Actual")
    else:
        ax3.text(0.5, 0.5, "No confusion matrix data", ha="center", va="center")
        ax3.set_title("Confusion Matrix")
        ax3.axis("off")

    # Chart 4: Top positive terms (aggregated across groups)
    ax4 = axes[1, 0]
    top_positive = flatten_top_terms(group_report, "top_positive_terms", args.top_n_terms)
    if top_positive:
        terms = [t for t, _ in top_positive][::-1]
        counts = [c for _, c in top_positive][::-1]
        ax4.barh(terms, counts, color="#2e8b57")
        ax4.set_title(f"Top {args.top_n_terms} Positive Terms (All Groups)")
        ax4.set_xlabel("Count")
    else:
        ax4.text(0.5, 0.5, "No positive term data", ha="center", va="center")
        ax4.set_title("Top Positive Terms")
        ax4.axis("off")

    # Chart 5: Top negative terms (aggregated across groups)
    ax5 = axes[1, 1]
    top_negative = flatten_top_terms(group_report, "top_negative_terms", args.top_n_terms)
    if top_negative:
        terms = [t for t, _ in top_negative][::-1]
        counts = [c for _, c in top_negative][::-1]
        ax5.barh(terms, counts, color="#b22222")
        ax5.set_title(f"Top {args.top_n_terms} Negative Terms (All Groups)")
        ax5.set_xlabel("Count")
    else:
        ax5.text(0.5, 0.5, "No negative term data", ha="center", va="center")
        ax5.set_title("Top Negative Terms")
        ax5.axis("off")

    # Chart 6: Sentiment distribution in predictions
    ax6 = axes[1, 2]
    if predictions_path.exists():
        predictions_df = pd.read_csv(predictions_path)
        if "predicted_sentiment" in predictions_df.columns:
            sentiment_counts = (
                predictions_df["predicted_sentiment"]
                .astype(str)
                .value_counts()
                .sort_values(ascending=False)
            )
            ax6.bar(
                sentiment_counts.index,
                sentiment_counts.values,
                color=["#1f77b4", "#ff7f0e", "#2ca02c"][: len(sentiment_counts)],
            )
            ax6.set_title("Predicted Sentiment Distribution")
            ax6.set_xlabel("Sentiment")
            ax6.set_ylabel("Document Count")
        else:
            ax6.text(0.5, 0.5, "Column predicted_sentiment missing", ha="center", va="center")
            ax6.set_title("Predicted Sentiment Distribution")
            ax6.axis("off")
    else:
        ax6.text(0.5, 0.5, "predictions.csv not found", ha="center", va="center")
        ax6.set_title("Predicted Sentiment Distribution")
        ax6.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"Saved dashboard to: {output_path}")


if __name__ == "__main__":
    main()
