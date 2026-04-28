#!/usr/bin/env python3
"""
DTM-SA Pipeline Runner

This script orchestrates the complete DTM-SA (Distributed Text Mining & Sentiment Analysis) workflow:
1. Run the sentiment analysis pipeline
2. Perform scalability testing
3. Validate results against ground truth
4. Generate comprehensive visualizations

Usage:
    python runner.py --input-csv ../data/test.csv --output-dir results

The script will create the following outputs:
- results/predictions.csv: Pipeline predictions
- results/group_term_sentiment_report.json: Term frequency analysis
- results/scalability_results.csv: Performance benchmarking
- results/validation_results.json: Quality validation metrics
- results/dashboard.png: Comprehensive visualization dashboard
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def run_command(cmd: List[str], description: str, cwd: Path = None) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)

    # Default to src directory if no cwd specified
    if cwd is None:
        cwd = Path(__file__).parent  # src directory

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if result.stdout.strip():
                print("Output:", result.stdout.strip())
            return True
        else:
            print(f"❌ {description} failed with exit code {result.returncode}")
            if result.stderr.strip():
                print("Error:", result.stderr.strip())
            return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run complete DTM-SA pipeline workflow"
    )
    parser.add_argument(
        "--input-csv",
        default="../data/test.csv",
        help="Path to input CSV file with text data"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Base directory for all outputs"
    )
    parser.add_argument(
        "--vectorizer-path",
        default="../model/models/tfidf_vectorizer.pkl",
        help="Path to trained TF-IDF vectorizer"
    )
    parser.add_argument(
        "--model-path",
        default="../model/models/sentiment_model.pkl",
        help="Path to trained sentiment model"
    )
    parser.add_argument(
        "--text-col",
        default="text",
        help="Name of text column in input CSV"
    )
    parser.add_argument(
        "--group-col",
        default="group",
        help="Name of group column in input CSV"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum number of documents to process (None = all)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker threads"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top terms to extract per group/sentiment"
    )
    parser.add_argument(
        "--scalability-doc-counts",
        default="100,500,1000",
        help="Document counts for scalability testing (comma-separated)"
    )
    parser.add_argument(
        "--scalability-chunk-sizes",
        default="500,1000,2000",
        help="Chunk sizes for scalability testing (comma-separated)"
    )
    parser.add_argument(
        "--skip-scalability",
        action="store_true",
        help="Skip scalability testing (faster execution)"
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Skip visualization generation"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert paths to be relative to src/ directory
    src_dir = Path(__file__).parent
    root_dir = src_dir.parent

    def make_src_relative(path_str: str) -> str:
        """Convert a path to be relative to src/ directory."""
        path = Path(path_str)
        if not path.is_absolute():
            # If path already starts with ../, it's already relative to src
            if path_str.startswith('../'):
                return path_str
            # Otherwise, assume it's relative to root and make it relative to src
            abs_path = (root_dir / path).resolve()
            try:
                return str(abs_path.relative_to(src_dir.resolve()))
            except ValueError:
                # Calculate relative path manually
                src_parts = src_dir.resolve().parts
                abs_parts = abs_path.parts

                # Find common prefix
                common_len = 0
                for i, (src_part, abs_part) in enumerate(zip(src_parts, abs_parts)):
                    if src_part == abs_part:
                        common_len = i + 1
                    else:
                        break

                # Go up from src to common ancestor
                up_levels = len(src_parts) - common_len
                down_parts = abs_parts[common_len:]

                relative_parts = ['..'] * up_levels + list(down_parts)
                return str(Path(*relative_parts))
        return path_str

    input_csv_rel = make_src_relative(args.input_csv)
    vectorizer_path_rel = make_src_relative(args.vectorizer_path)
    model_path_rel = make_src_relative(args.model_path)
    output_dir_rel = make_src_relative(args.output_dir)

    print("🚀 Starting DTM-SA Pipeline Runner")
    print(f"Input: {args.input_csv} (relative to src/: {input_csv_rel})")
    print(f"Output directory: {args.output_dir} (relative to src/: {output_dir_rel})")

    # Step 1: Run the main pipeline
    pipeline_cmd = [
        sys.executable, "run_pipeline.py",
        "--input-csv", input_csv_rel,
        "--output-dir", output_dir_rel,
        "--vectorizer-path", vectorizer_path_rel,
        "--model-path", model_path_rel,
        "--text-col", args.text_col,
        "--group-col", args.group_col,
        "--chunk-size", str(args.chunk_size),
        "--top-n", str(args.top_n),
        "--workers", str(args.workers)
    ]

    if args.max_docs is not None:
        pipeline_cmd.extend(["--max-docs", str(args.max_docs)])

    if not run_command(pipeline_cmd, "Sentiment Analysis Pipeline"):
        print("❌ Pipeline failed. Stopping execution.")
        return 1

    # Step 2: Run scalability testing (optional)
    if not args.skip_scalability:
        scalability_cmd = [
            sys.executable, "scalability.py",
            "--input-csv", input_csv_rel,
            "--output-csv", f"{output_dir_rel}/scalability_results.csv",
            "--output-dir", f"{output_dir_rel}/scalability_runs",
            "--vectorizer-path", vectorizer_path_rel,
            "--model-path", model_path_rel,
            "--text-col", args.text_col,
            "--group-col", args.group_col,
            "--doc-counts", args.scalability_doc_counts,
            "--chunk-sizes", args.scalability_chunk_sizes,
            "--workers", str(args.workers),
            "--top-n", str(args.top_n)
        ]

        if not run_command(scalability_cmd, "Scalability Testing"):
            print("⚠️  Scalability testing failed, but continuing...")
    else:
        print("⏭️  Skipping scalability testing")

    # Step 3: Run validation
    validation_cmd = [
        sys.executable, "validate.py",
        "--input-csv", input_csv_rel,
        "--predictions-csv", f"{output_dir_rel}/predictions.csv",
        "--output-json", f"{output_dir_rel}/validation_results.json"
    ]

    if not run_command(validation_cmd, "Results Validation"):
        print("⚠️  Validation failed, but continuing...")

    # Step 4: Generate visualizations (optional)
    if not args.skip_visualization:
        visualization_cmd = [
            sys.executable, "visualize_results.py",
            "--scalability-csv", f"{output_dir_rel}/scalability_results.csv",
            "--validation-json", f"{output_dir_rel}/validation_results.json",
            "--group-report-json", f"{output_dir_rel}/group_term_sentiment_report.json",
            "--predictions-csv", f"{output_dir_rel}/predictions.csv",
            "--output-png", f"{output_dir_rel}/dashboard.png"
        ]

        if not run_command(visualization_cmd, "Visualization Generation"):
            print("⚠️  Visualization failed, but continuing...")
    else:
        print("⏭️  Skipping visualization generation")

    # Summary
    print(f"\n{'='*60}")
    print("🎉 DTM-SA Pipeline Runner completed!")
    print(f"{'='*60}")
    print(f"📁 All outputs saved to: {output_dir.absolute()}")
    print("\n📊 Generated files:")
    print(f"  • predictions.csv - Sentiment predictions")
    print(f"  • group_term_sentiment_report.json - Term frequency analysis")
    print(f"  • pipeline_runtime.json - Performance metrics")

    if not args.skip_scalability:
        print(f"  • scalability_results.csv - Performance benchmarking")

    print(f"  • validation_results.json - Quality validation metrics")

    if not args.skip_visualization:
        print(f"  • dashboard.png - Comprehensive visualization dashboard")

    print(f"\n💡 Next steps:")
    print(f"  • View dashboard.png for visual analysis")
    print(f"  • Check validation_results.json for model quality metrics")
    print(f"  • Review scalability_results.csv for performance insights")

    return 0


if __name__ == "__main__":
    sys.exit(main())