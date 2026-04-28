#format_metrics.py
#reads metrics.json (+ optional runtime/terms jsons) and prints a clean report
#usage: python format_metrics.py [--metrics metrics.json] [--runtime pipeline_runtime.json] [--terms group_term_sentiment_report.json]

import argparse
import json
import os
from datetime import datetime


def load_json(path: str):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def pct(val: float) -> str:
    return f"{val * 100:.2f}%"


def fmt(val: float, decimals: int = 3) -> str:
    return f"{val:.{decimals}f}"


def section_classification(metrics: dict) -> tuple[str, str]:
    acc = metrics["accuracy"]
    report = metrics["classification_report"]
    cm = metrics["confusion_matrix"]
    train_size = metrics.get("train_size", "N/A")
    test_size = metrics.get("test_size", "N/A")
    cap = metrics.get("max_samples_per_group", "N/A")
    classes = ["negative", "neutral", "positive"]

    lines = []
    lines.append("=" * 60)
    lines.append("SENTIMENT CLASSIFICATION PERFORMANCE")
    lines.append("=" * 60)
    lines.append(f"  Training samples : {train_size:,}" if isinstance(train_size, int) else f"  Training samples : {train_size}")
    lines.append(f"  Test samples     : {test_size:,}" if isinstance(test_size, int) else f"  Test samples     : {test_size}")
    lines.append(f"  Max per class    : {cap:,}" if isinstance(cap, int) else f"  Max per class    : {cap}")
    lines.append(f"  Overall accuracy : {pct(acc)}")
    lines.append("")
    lines.append(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    lines.append(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for cls in classes:
        r = report[cls]
        lines.append(f"  {cls:<12} {fmt(r['precision']):>10} {fmt(r['recall']):>10} " f"{fmt(r['f1-score']):>10} {int(r['support']):>10}")
    lines.append(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    ma = report["macro avg"]
    wa = report["weighted avg"]
    lines.append(
        f"  {'macro avg':<12} {fmt(ma['precision']):>10} {fmt(ma['recall']):>10} "
        f"{fmt(ma['f1-score']):>10} {int(ma['support']):>10}"
    )
    lines.append(
        f"  {'weighted avg':<12} {fmt(wa['precision']):>10} {fmt(wa['recall']):>10} "
        f"{fmt(wa['f1-score']):>10} {int(wa['support']):>10}"
    )
    lines.append("")
    lines.append("  Confusion Matrix (rows = actual, cols = predicted)")
    lines.append(f"  {'':>16} {'neg':>8} {'neu':>8} {'pos':>8}")
    for i, lbl in enumerate(classes):
        lines.append(f"  {'actual ' + lbl:<16} {cm[i][0]:>8} {cm[i][1]:>8} {cm[i][2]:>8}")
    plain = "\n".join(lines)

    md = []
    md.append("## Sentiment Classification Performance\n")
    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    md.append(f"| Training samples | {train_size:,} |" if isinstance(train_size, int) else f"| Training samples | {train_size} |")
    md.append(f"| Test samples | {test_size:,} |" if isinstance(test_size, int) else f"| Test samples | {test_size} |")
    md.append(f"| Max samples per class | {cap:,} |" if isinstance(cap, int) else f"| Max samples per class | {cap} |")
    md.append(f"| Overall accuracy | **{pct(acc)}** |\n")
    md.append("### Per-class Metrics\n")
    md.append("| Class | Precision | Recall | F1-score | Support |")
    md.append("|-------|-----------|--------|----------|---------|")
    for cls in classes:
        r = report[cls]
        md.append(
            f"| {cls} | {fmt(r['precision'])} | {fmt(r['recall'])} | "
            f"**{fmt(r['f1-score'])}** | {int(r['support'])} |"
        )
    md.append(f"| macro avg | {fmt(ma['precision'])} | {fmt(ma['recall'])} | {fmt(ma['f1-score'])} | {int(ma['support'])} |")
    md.append(f"| weighted avg | {fmt(wa['precision'])} | {fmt(wa['recall'])} | {fmt(wa['f1-score'])} | {int(wa['support'])} |")
    md.append("\n### Confusion Matrix\n")
    md.append("| | Pred. Negative | Pred. Neutral | Pred. Positive |")
    md.append("|--|----------------|---------------|----------------|")
    for i, lbl in enumerate(classes):
        md.append(f"| **Actual {lbl}** | {cm[i][0]} | {cm[i][1]} | {cm[i][2]} |")

    return plain, "\n".join(md)


def section_top_features(metrics: dict) -> tuple[str, str]:
    top = metrics.get("top_features", {})
    if not top:
        return "", ""

    # filter out generic terms that don't say much
    skip = {"and", "not", "but", "or", "the", "is"}

    lines = []
    lines.append("=" * 60)
    lines.append("TOP CLASSIFIER FEATURES PER SENTIMENT CLASS")
    lines.append("=" * 60)

    md = ["## Top Classifier Features per Sentiment Class\n"]

    for cls in ["negative", "neutral", "positive"]:
        if cls not in top:
            continue
        feats = [f for f in top[cls]["top_positive_features"] if f["term"] not in skip][:10]

        lines.append(f"\n  [{cls.upper()}]")
        lines.append(f"  {'Term':<20} {'Weight':>8}")
        lines.append(f"  {'-'*20} {'-'*8}")
        for feat in feats:
            lines.append(f"  {feat['term']:<20} {feat['weight']:>8.3f}")

        md.append(f"### {cls.capitalize()}\n")
        md.append("| Term | Weight |")
        md.append("|------|--------|")
        for feat in feats:
            md.append(f"| {feat['term']} | {feat['weight']:.3f} |")
        md.append("")

    return "\n".join(lines), "\n".join(md)


def section_runtime(data) -> tuple[str, str]:
    if not data:
        return "", ""

    #handle both list and dict formats
    if isinstance(data, list):
        records = data
    elif "groups" in data:
        records = data["groups"]
    else:
        records = [data]

    lines = ["=" * 60, "PIPELINE RUNTIME", "=" * 60]
    md = ["## Pipeline Runtime\n", "| Group | Documents | Time (s) |", "|-------|-----------|----------|"]

    total = 0.0
    for rec in records:
        group = rec.get("group", "unknown")
        docs = rec.get("doc_count", rec.get("documents", "N/A"))
        t = rec.get("execution_time_s", rec.get("time_s", 0.0))
        total += t if isinstance(t, (int, float)) else 0
        lines.append(f"  {group:<14} {str(docs):>10}    {t:.2f}s")
        md.append(f"| {group} | {docs} | {t:.2f} |")

    if total:
        lines.append(f"\n  Total: {total:.2f}s")
        md.append(f"\n**Total:** {total:.2f}s")

    return "\n".join(lines), "\n".join(md)


def section_terms(data) -> tuple[str, str]:
    if not data:
        return "", ""

    records = data if isinstance(data, list) else [data]

    lines = ["=" * 60, "TOP TERMS PER GROUP AND SENTIMENT", "=" * 60]
    md = ["## Top Terms per Group and Sentiment\n"]

    for rec in records:
        group = rec.get("group", "unknown")
        sentiment = rec.get("sentiment", "unknown")
        terms = rec.get("top_terms", [])
        lines.append(f"\n  {group.upper()} / {sentiment}")
        lines.append(f"  Terms: {', '.join(terms)}")
        md.append(f"### {group.capitalize()} - {sentiment.capitalize()}\n")
        md.append(f"Top terms: {', '.join(f'`{t}`' for t in terms)}\n")

    return "\n".join(lines), "\n".join(md)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="metrics.json")
    parser.add_argument("--runtime", default="pipeline_runtime.json")
    parser.add_argument("--terms", default="group_term_sentiment_report.json")
    args = parser.parse_args()

    metrics = load_json(args.metrics)
    if metrics is None:
        print(f"ERROR: '{args.metrics}' not found")
        return

    runtime = load_json(args.runtime)
    terms = load_json(args.terms)

    os.makedirs("reports", exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    plain_parts = [f"Pipeline Metrics Report  |  {ts}\n"]
    md_parts = [f"# Pipeline Metrics Report\n\n_{ts}_\n"]

    for fn, data in [
        (section_classification, metrics),
        (section_top_features, metrics),
        (section_runtime, runtime),
        (section_terms, terms),
    ]:
        p, m = fn(data)
        if p:
            plain_parts.append(p)
        if m:
            md_parts.append(m)

    plain = "\n\n".join(plain_parts)
    md = "\n\n".join(md_parts)

    with open("reports/metrics_summary.txt", "w") as f:
        f.write(plain)
    with open("reports/metrics_summary.md", "w") as f:
        f.write(md)

    print(plain)
    print("\nsaved to reports/metrics_summary.txt and reports/metrics_summary.md")


if __name__ == "__main__":
    main()
