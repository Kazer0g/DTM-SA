import argparse
import csv
import time
from pathlib import Path
from typing import List

from pipeline import run_pipeline


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Run scalability experiments.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", default="outputs/scalability_results.csv")
    parser.add_argument("--output-dir", default="outputs/scalability_runs")
    parser.add_argument("--vectorizer-path", default="../tfidf_vectorizer.pkl")
    parser.add_argument("--model-path", default="../sentiment_model.pkl")
    parser.add_argument("--label-encoder-path", default="../label_encoder.pkl")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--group-col", default="group")
    parser.add_argument("--doc-counts", default="1000,10000,100000")
    parser.add_argument("--chunk-sizes", default="500,2000,5000,10000")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    doc_counts = parse_int_list(args.doc_counts)
    chunk_sizes = parse_int_list(args.chunk_sizes)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for max_docs in doc_counts:
        baseline_time = None
        for chunk_size in chunk_sizes:
            run_output_dir = base_output_dir / f"docs_{max_docs}_chunk_{chunk_size}"
            start = time.perf_counter()
            summary = run_pipeline(
                input_csv=Path(args.input_csv),
                output_dir=run_output_dir,
                vectorizer_path=Path(args.vectorizer_path),
                model_path=Path(args.model_path),
                label_encoder_path=Path(args.label_encoder_path),
                text_col=args.text_col,
                group_col=args.group_col,
                chunksize=chunk_size,
                max_docs=max_docs,
                top_n=args.top_n,
                workers=args.workers,
            )
            elapsed = time.perf_counter() - start
            if baseline_time is None:
                baseline_time = elapsed
            speedup_vs_first_chunk = baseline_time / elapsed if elapsed > 0 else None
            rows.append(
                {
                    "max_docs": max_docs,
                    "chunk_size": chunk_size,
                    "workers": args.workers,
                    "elapsed_seconds": elapsed,
                    "processed_docs": summary["processed_docs"],
                    "speedup_vs_first_chunk": speedup_vs_first_chunk,
                }
            )

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "max_docs",
                "chunk_size",
                "workers",
                "elapsed_seconds",
                "processed_docs",
                "speedup_vs_first_chunk",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved scalability results to {output_csv}")


if __name__ == "__main__":
    main()
