import json
import re
import time
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
from pyspark import StorageLevel
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (
    ArrayType,
    LongType,
    StringType,
    StructField,
    StructType,
)
from pyspark.sql.window import Window
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

TOKEN_PATTERN = re.compile(r"[A-Za-z']+")
ESCAPED_SEQUENCE_PATTERN = re.compile(r"\\[nrtfvba0]+")
ESCAPED_UNICODE_PATTERN = re.compile(r"\\u[0-9a-fA-F]{4}")
SENTIMENTS_FOR_TOP = ("positive", "neutral", "negative")
EXTRA_STOP_WORDS = {"anything", "something", "everything", "really", "just", "also"}
# Category prefixes are intentionally added during model training to improve
# classification, but they should not dominate "top terms" analytics outputs.
DOMAIN_STOP_WORDS = {"movies", "movie", "restaurants", "restaurant", "tweets", "tweet"}
# Escaped newlines in CSV text (e.g. "\\nI") can produce token artifact "ni".
ARTIFACT_STOP_WORDS = {"ni"}
STOP_WORDS = set(ENGLISH_STOP_WORDS) | EXTRA_STOP_WORDS
TERM_EXCLUDE_WORDS = STOP_WORDS | DOMAIN_STOP_WORDS | ARTIFACT_STOP_WORDS


def tokenize(text: str) -> List[str]:
    """Tokenize text for term analytics (not for model inference)."""
    if not isinstance(text, str):
        return []
    # Remove escaped control/unicode sequences from CSV text (e.g. "\\nI", "\\u00f1")
    # so artifacts after backslash escapes are not treated as terms.
    normalized_text = ESCAPED_SEQUENCE_PATTERN.sub(" ", text)
    normalized_text = ESCAPED_UNICODE_PATTERN.sub(" ", normalized_text)
    normalized_text = normalized_text.replace("\\", " ")

    cleaned_tokens: List[str] = []
    for token in TOKEN_PATTERN.findall(normalized_text.lower()):
        normalized = token.strip("'")
        if len(normalized) <= 1:
            continue
        if normalized in TERM_EXCLUDE_WORDS:
            continue
        cleaned_tokens.append(normalized)
    return cleaned_tokens


def load_artifacts(
    vectorizer_path: Path, model_path: Path
):
    """Load pre-trained vectorizer + classifier artifacts once per worker."""
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    return vectorizer, model


def _predict_partition(
    rows: Iterable[Tuple[int, str, str]],
    vectorizer_path: str,
    model_path: str,
    chunk_size: int,
):
    """
    MAP phase (inference):
    - each partition loads artifacts once
    - rows are batched and mapped to predicted sentiment labels
    """
    vectorizer, model = load_artifacts(
        Path(vectorizer_path), Path(model_path)
    )
    buffered_row_ids: List[int] = []
    buffered_texts: List[str] = []
    buffered_groups: List[str] = []

    def flush_buffer():
        if not buffered_texts:
            return []
        features = vectorizer.transform(buffered_texts)
        pred_labels = model.predict(features)
        output = [
            (int(row_id), group_name, text, str(sentiment))
            for row_id, group_name, text, sentiment in zip(
                buffered_row_ids, buffered_groups, buffered_texts, pred_labels
            )
        ]
        buffered_row_ids.clear()
        buffered_texts.clear()
        buffered_groups.clear()
        return output

    for row_id, text, group_name in rows:
        buffered_row_ids.append(int(row_id))
        buffered_texts.append("" if text is None else str(text))
        buffered_groups.append("unknown" if group_name is None else str(group_name))

        if len(buffered_texts) >= chunk_size:
            for item in flush_buffer():
                yield item

    for item in flush_buffer():
        yield item


def reduce_term_counts_spark(predictions_df, top_n: int) -> Dict[str, List[Dict[str, int]]]:
    """
    REDUCE phase (term analytics):
    1) map each document to tokens
    2) reduce by (sentiment, term) with count aggregation
    3) rank and keep top N terms per sentiment globally
    """
    tokenize_udf = F.udf(tokenize, ArrayType(StringType()))
    terms_df = (
        predictions_df.filter(F.col("predicted_sentiment").isin(*SENTIMENTS_FOR_TOP))
        .withColumn("term", F.explode(tokenize_udf(F.col("text"))))
        .groupBy("predicted_sentiment", "term")
        .count()
    )

    ranking_window = Window.partitionBy("predicted_sentiment").orderBy(
        F.col("count").desc(), F.col("term").asc()
    )
    top_df = terms_df.withColumn("rank", F.row_number().over(ranking_window)).filter(
        F.col("rank") <= top_n
    )

    # For each (sentiment, term), keep up to 5 example messages where this term appears.
    examples_window = Window.partitionBy("predicted_sentiment", "term").orderBy(F.col("row_id").asc())
    term_examples_df = (
        predictions_df.filter(F.col("predicted_sentiment").isin(*SENTIMENTS_FOR_TOP))
        .withColumn("term", F.explode(tokenize_udf(F.col("text"))))
        .select("row_id", "predicted_sentiment", "term", "text")
        .dropDuplicates(["row_id", "predicted_sentiment", "term", "text"])
        .withColumn("example_rank", F.row_number().over(examples_window))
        .filter(F.col("example_rank") <= 5)
        .groupBy("predicted_sentiment", "term")
        .agg(
            F.collect_list(
                F.struct(
                    F.col("text").alias("message"),
                    F.col("predicted_sentiment").alias("sentiment_tag"),
                )
            ).alias("examples")
        )
    )

    top_with_examples_df = top_df.join(
        term_examples_df,
        on=["predicted_sentiment", "term"],
        how="left",
    )

    result: Dict[str, List[Dict[str, int]]] = {
        f"top_{sentiment}_terms": [] for sentiment in SENTIMENTS_FOR_TOP
    }
    for row in top_with_examples_df.collect():
        sentiment = str(row["predicted_sentiment"])
        examples = []
        for item in row["examples"] or []:
            examples.append(
                {
                    "message": str(item["message"]),
                    "sentiment_tag": str(item["sentiment_tag"]),
                }
            )
        result[f"top_{sentiment}_terms"].append(
            {
                "term": str(row["term"]),
                "count": int(row["count"]),
                "examples": examples,
            }
        )

    return result


def compute_work_statistics(predictions_df) -> Dict[str, object]:
    """Compute additional aggregate statistics from prediction outputs."""
    tokenize_udf = F.udf(tokenize, ArrayType(StringType()))
    sentiment_df = predictions_df.filter(F.col("predicted_sentiment").isin(*SENTIMENTS_FOR_TOP))

    # Map phase: tokenize and emit term counts per document
    tokenized = sentiment_df.select(
        "row_id",
        "group",
        "predicted_sentiment",
        tokenize_udf(F.col("text")).alias("terms")
    )

    exploded = tokenized.withColumn("term", F.explode(F.col("terms"))).select(
        "row_id", "group", "predicted_sentiment", "term"
    )

    term_counts = exploded.groupBy("predicted_sentiment", "term").count()
    total_terms = term_counts.groupBy("predicted_sentiment").agg(
        F.sum("count").alias("total_term_count")
    )
    distinct_terms = term_counts.groupBy("predicted_sentiment").count().withColumnRenamed("count", "distinct_term_count")

    doc_counts = sentiment_df.groupBy("predicted_sentiment").count().withColumnRenamed("count", "document_count")
    terms_per_doc = tokenized.select(
        "predicted_sentiment", F.size(F.col("terms")).alias("tokens_per_doc")
    ).groupBy("predicted_sentiment").agg(
        F.avg("tokens_per_doc").alias("average_tokens_per_doc"),
        F.min("tokens_per_doc").alias("min_tokens_per_doc"),
        F.max("tokens_per_doc").alias("max_tokens_per_doc"),
    )

    joined = doc_counts.join(total_terms, on="predicted_sentiment", how="left").join(
        distinct_terms, on="predicted_sentiment", how="left").join(
        terms_per_doc, on="predicted_sentiment", how="left"
    )

    stats = {
        str(row["predicted_sentiment"]): {
            "document_count": int(row["document_count"]),
            "total_term_count": int(row["total_term_count"] or 0),
            "distinct_term_count": int(row["distinct_term_count"] or 0),
            "average_tokens_per_doc": float(row["average_tokens_per_doc"] or 0.0),
            "min_tokens_per_doc": int(row["min_tokens_per_doc"] or 0),
            "max_tokens_per_doc": int(row["max_tokens_per_doc"] or 0),
        }
        for row in joined.collect()
    }

    top_terms_overall = term_counts.orderBy(F.col("count").desc(), F.col("term").asc()).limit(50).collect()
    overall_terms = [
        {"term": str(row["term"]), "sentiment": str(row["predicted_sentiment"]), "count": int(row["count"])}
        for row in top_terms_overall
    ]

    return {
        "sentiment_statistics": stats,
        "top_terms_overall": overall_terms,
    }


def run_pipeline(
    input_csv: Path,
    output_dir: Path,
    vectorizer_path: Path,
    model_path: Path,
    text_col: str = "text",
    group_col: str = "group",
    chunksize: int = 2000,
    max_docs: Optional[int] = None,
    top_n: int = 20,
    workers: int = 1,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    spark = (
        SparkSession.builder.appName("DTM-Sentiment-Pipeline")
        .master(f"local[{max(1, workers)}]")
        .getOrCreate()
    )
    try:
        # Keep CSV parsing aligned with pandas behavior for quoted multiline text.
        source_df = (
            spark.read
            .option("header", True)
            .option("multiLine", True)
            .option("quote", "\"")
            .option("escape", "\"")
            .csv(str(input_csv))
        )
        if group_col in source_df.columns:
            selected_df = source_df.select(
                F.coalesce(F.col(text_col).cast("string"), F.lit("")).alias("text"),
                F.coalesce(F.col(group_col).cast("string"), F.lit("unknown")).alias("group"),
            )
        else:
            selected_df = source_df.select(
                F.coalesce(F.col(text_col).cast("string"), F.lit("")).alias("text"),
                F.lit("unknown").alias("group"),
            )

        if max_docs is not None:
            selected_df = selected_df.limit(int(max_docs))

        indexed_rdd = selected_df.rdd.zipWithIndex().map(
            lambda pair: (int(pair[1]), pair[0]["text"], pair[0]["group"])
        )

        predicted_rdd = indexed_rdd.mapPartitions(
            lambda rows: _predict_partition(
                rows=rows,
                vectorizer_path=str(vectorizer_path),
                model_path=str(model_path),
                chunk_size=max(1, int(chunksize)),
            )
        )

        predictions_schema = StructType(
            [
                StructField("row_id", LongType(), nullable=False),
                StructField("group", StringType(), nullable=False),
                StructField("text", StringType(), nullable=False),
                StructField("predicted_sentiment", StringType(), nullable=False),
            ]
        )
        predictions_df = spark.createDataFrame(predicted_rdd, schema=predictions_schema)
        predictions_df = predictions_df.persist(StorageLevel.MEMORY_AND_DISK)
        processed_total = predictions_df.count()

        grouped_report = reduce_term_counts_spark(predictions_df=predictions_df, top_n=top_n)
        work_statistics = compute_work_statistics(predictions_df=predictions_df)
        prediction_rows = [
            {
                "row_id": str(row["row_id"]),
                "group": row["group"],
                "text": row["text"],
                "predicted_sentiment": row["predicted_sentiment"],
            }
            for row in predictions_df.orderBy("row_id").toLocalIterator()
        ]
    finally:
        spark.stop()

    predictions_path = output_dir / "predictions.csv"
    grouped_path = output_dir / "group_term_sentiment_report.json"
    work_stats_path = output_dir / "work_statistics.json"
    runtime_path = output_dir / "pipeline_runtime.json"

    with predictions_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=["row_id", "group", "text", "predicted_sentiment"],
            quoting=csv.QUOTE_ALL
        )
        writer.writeheader()
        writer.writerows(prediction_rows)
    grouped_path.write_text(json.dumps(grouped_report, indent=2), encoding="utf-8")
    work_stats_path.write_text(json.dumps(work_statistics, indent=2), encoding="utf-8")

    elapsed_seconds = time.perf_counter() - start_time
    runtime_summary = {
        "processed_docs": processed_total,
        "chunksize": chunksize,
        "workers": workers,
        "top_n": top_n,
        "elapsed_seconds": elapsed_seconds,
        "predictions_path": str(predictions_path),
        "group_report_path": str(grouped_path),
        "work_statistics_path": str(work_stats_path),
    }
    runtime_path.write_text(json.dumps(runtime_summary, indent=2), encoding="utf-8")
    return runtime_summary
