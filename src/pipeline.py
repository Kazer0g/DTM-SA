import json
import re
import time
import csv
from collections import defaultdict
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
SENTIMENTS_FOR_TOP = ("positive", "negative")
STOP_WORDS = set(ENGLISH_STOP_WORDS)


def tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return [
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if len(token) > 1 and token not in STOP_WORDS
    ]


def load_artifacts(
    vectorizer_path: Path, model_path: Path
):
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    return vectorizer, model


def _predict_partition(
    rows: Iterable[Tuple[int, str, str]],
    vectorizer_path: str,
    model_path: str,
    chunk_size: int,
):
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


def reduce_term_counts_spark(predictions_df, top_n: int) -> Dict[str, Dict[str, List[Dict[str, int]]]]:
    tokenize_udf = F.udf(tokenize, ArrayType(StringType()))
    terms_df = (
        predictions_df.filter(F.col("predicted_sentiment").isin(*SENTIMENTS_FOR_TOP))
        .withColumn("term", F.explode(tokenize_udf(F.col("text"))))
        .groupBy("group", "predicted_sentiment", "term")
        .count()
    )

    ranking_window = Window.partitionBy("group", "predicted_sentiment").orderBy(
        F.col("count").desc(), F.col("term").asc()
    )
    top_df = terms_df.withColumn("rank", F.row_number().over(ranking_window)).filter(
        F.col("rank") <= top_n
    )

    result: Dict[str, Dict[str, List[Dict[str, int]]]] = defaultdict(
        lambda: {f"top_{sentiment}_terms": [] for sentiment in SENTIMENTS_FOR_TOP}
    )
    for row in top_df.collect():
        group_name = str(row["group"])
        sentiment = str(row["predicted_sentiment"])
        result[group_name][f"top_{sentiment}_terms"].append(
            {"term": str(row["term"]), "count": int(row["count"])}
        )

    return dict(result)


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
        source_df = spark.read.option("header", True).csv(str(input_csv))
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
    runtime_path = output_dir / "pipeline_runtime.json"

    with predictions_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=["row_id", "group", "text", "predicted_sentiment"],
            quoting=csv.QUOTE_ALL
        )
        writer.writeheader()
        writer.writerows(prediction_rows)
    grouped_path.write_text(json.dumps(grouped_report, indent=2), encoding="utf-8")

    elapsed_seconds = time.perf_counter() - start_time
    runtime_summary = {
        "processed_docs": processed_total,
        "chunksize": chunksize,
        "workers": workers,
        "top_n": top_n,
        "elapsed_seconds": elapsed_seconds,
        "predictions_path": str(predictions_path),
        "group_report_path": str(grouped_path),
    }
    runtime_path.write_text(json.dumps(runtime_summary, indent=2), encoding="utf-8")
    return runtime_summary
