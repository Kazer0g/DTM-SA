import argparse
import json
from pathlib import Path

from pipeline import run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run local MapReduce-style sentiment + term mining pipeline."
    )
    parser.add_argument("--input-csv", required=True, help="Path to input CSV.")
    parser.add_argument(
        "--output-dir", default="outputs", help="Directory for output artifacts."
    )
    parser.add_argument(
        "--vectorizer-path",
        default="../model/models/tfidf_vectorizer.pkl",
        help="Path to TF-IDF vectorizer artifact.",
    )
    parser.add_argument(
        "--model-path",
        default="../model/models/sentiment_model.pkl",
        help="Path to sentiment model artifact.",
    )
    parser.add_argument("--text-col", default="text", help="Text column name.")
    parser.add_argument("--group-col", default="group", help="Group column name.")
    parser.add_argument("--chunk-size", type=int, default=2000, help="CSV chunksize.")
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional cap on number of documents to process.",
    )
    parser.add_argument("--top-n", type=int, default=20, help="Top terms per class/group.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for map phase.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary = run_pipeline(
        input_csv=Path(args.input_csv),
        output_dir=Path(args.output_dir),
        vectorizer_path=Path(args.vectorizer_path),
        model_path=Path(args.model_path),
        text_col=args.text_col,
        group_col=args.group_col,
        chunksize=args.chunk_size,
        max_docs=args.max_docs,
        top_n=args.top_n,
        workers=args.workers,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
