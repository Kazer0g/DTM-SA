from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import pandas as pd

TOKEN_PATTERN = re.compile(r"[A-Za-z']+")
SENTIMENTS_FOR_TOP = ("positive", "negative")


@dataclass
class ChunkResult:
    prediction_rows: List[Dict[str, str]]
    term_counts: Dict[Tuple[str, str, str], int]
    processed_rows: int


def tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return [token for token in TOKEN_PATTERN.findall(text.lower()) if len(token) > 1]


def load_artifacts(
    vectorizer_path: Path, model_path: Path, label_encoder_path: Path
):
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)
    return vectorizer, model, label_encoder


def map_chunk(
    chunk_df: pd.DataFrame,
    vectorizer,
    model,
    label_encoder,
    text_col: str,
    group_col: str,
) -> ChunkResult:
    clean_chunk = chunk_df[[text_col, group_col]].copy()
    clean_chunk[text_col] = clean_chunk[text_col].fillna("").astype(str)
    clean_chunk[group_col] = clean_chunk[group_col].fillna("unknown").astype(str)

    features = vectorizer.transform(clean_chunk[text_col].tolist())
    pred_ids = model.predict(features)
    pred_labels = label_encoder.inverse_transform(pred_ids)

    prediction_rows: List[Dict[str, str]] = []
    term_counter: Counter = Counter()

    for idx, row in clean_chunk.iterrows():
        text = row[text_col]
        group_name = row[group_col]
        sentiment = str(pred_labels[len(prediction_rows)])

        prediction_rows.append(
            {
                "row_id": str(idx),
                "group": group_name,
                "text": text,
                "predicted_sentiment": sentiment,
            }
        )

        for term in tokenize(text):
            term_counter[(group_name, sentiment, term)] += 1

    return ChunkResult(
        prediction_rows=prediction_rows,
        term_counts=dict(term_counter),
        processed_rows=len(clean_chunk),
    )


def reduce_term_counts(
    mapped_outputs: Iterable[ChunkResult], top_n: int
) -> Dict[str, Dict[str, List[Dict[str, int]]]]:
    aggregate_counter: Counter = Counter()
    for item in mapped_outputs:
        aggregate_counter.update(item.term_counts)

    grouped: Dict[str, Dict[str, Counter]] = defaultdict(
        lambda: {sentiment: Counter() for sentiment in SENTIMENTS_FOR_TOP}
    )
    for (group_name, sentiment, term), value in aggregate_counter.items():
        if sentiment in SENTIMENTS_FOR_TOP:
            grouped[group_name][sentiment][term] += value

    result: Dict[str, Dict[str, List[Dict[str, int]]]] = {}
    for group_name, sentiment_counters in grouped.items():
        result[group_name] = {}
        for sentiment, term_counter in sentiment_counters.items():
            top_terms = term_counter.most_common(top_n)
            result[group_name][f"top_{sentiment}_terms"] = [
                {"term": term, "count": int(count)} for term, count in top_terms
            ]
    return result


def _map_chunk_worker(payload):
    chunk_df, vectorizer_path, model_path, label_encoder_path, text_col, group_col = payload
    vectorizer, model, label_encoder = load_artifacts(
        Path(vectorizer_path), Path(model_path), Path(label_encoder_path)
    )
    return map_chunk(
        chunk_df=chunk_df,
        vectorizer=vectorizer,
        model=model,
        label_encoder=label_encoder,
        text_col=text_col,
        group_col=group_col,
    )


def run_pipeline(
    input_csv: Path,
    output_dir: Path,
    vectorizer_path: Path,
    model_path: Path,
    label_encoder_path: Path,
    text_col: str = "text",
    group_col: str = "group",
    chunksize: int = 2000,
    max_docs: Optional[int] = None,
    top_n: int = 20,
    workers: int = 1,
) -> Dict[str, object]:

    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    prediction_rows: List[Dict[str, str]] = []
    mapped_outputs: List[ChunkResult] = []
    processed_total = 0

    reader = pd.read_csv(input_csv, chunksize=chunksize)

    if workers > 1:
        payloads = []
        for chunk in reader:
            if max_docs is not None and processed_total >= max_docs:
                break
            if max_docs is not None:
                remaining = max_docs - processed_total
                chunk = chunk.head(remaining)
            processed_total += len(chunk)
            payloads.append(
                (
                    chunk,
                    str(vectorizer_path),
                    str(model_path),
                    str(label_encoder_path),
                    text_col,
                    group_col,
                )
            )

        with ProcessPoolExecutor(max_workers=workers) as executor:
            for mapped in executor.map(_map_chunk_worker, payloads):
                mapped_outputs.append(mapped)
                prediction_rows.extend(mapped.prediction_rows)
    else:
        vectorizer, model, label_encoder = load_artifacts(
            vectorizer_path=vectorizer_path,
            model_path=model_path,
            label_encoder_path=label_encoder_path,
        )
        for chunk in reader:
            if max_docs is not None and processed_total >= max_docs:
                break
            if max_docs is not None:
                remaining = max_docs - processed_total
                chunk = chunk.head(remaining)
            mapped = map_chunk(
                chunk_df=chunk,
                vectorizer=vectorizer,
                model=model,
                label_encoder=label_encoder,
                text_col=text_col,
                group_col=group_col,
            )
            mapped_outputs.append(mapped)
            prediction_rows.extend(mapped.prediction_rows)
            processed_total += mapped.processed_rows

    grouped_report = reduce_term_counts(mapped_outputs, top_n=top_n)

    predictions_path = output_dir / "predictions.csv"
    grouped_path = output_dir / "group_term_sentiment_report.json"
    runtime_path = output_dir / "pipeline_runtime.json"

    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
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
