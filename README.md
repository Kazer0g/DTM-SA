# DTM-SA: Distributed Text Mining & Sentiment Analysis

## Project Overview

DTM-SA is a modular project for distributed text mining and sentiment analysis. It combines PySpark for scalable data processing with scikit-learn model training and evaluation. The repository includes model training, a distributed inference pipeline, validation, scalability testing, and visualization.

## Repository Structure

### Root Files

- `README.md` - Project overview, setup, and usage guide.
- `REPORT.md` - Project report and documentation.
- `requirements.txt` - Python dependencies.
- `.gitignore` - Files excluded from version control.
- `.venv/` - Optional Python virtual environment directory.

### `data/`

Contains curated input datasets for training and evaluation:

- `train.csv` - Training dataset used by `model/train_model.py`.
- `test.csv` - Evaluation dataset used by the pipeline and validation.

### `docs/`

Documentation and project guidelines:

- `GUIDE.md` - Developer guide and workflow best practices.
- `PROJECT-RULES.md` - Project rules and issue management guidelines.

### `model/`

Contains model training scripts, inference examples, and saved artifacts:

- `train_model.py` - Loads datasets, trains the sentiment model, and saves artifacts.
- `predictions.py` - Demonstrates how to load the saved model and predict sentiment for new text.
- `models/` - Saved model artifacts:
  - `tfidf_vectorizer.pkl`
  - `sentiment_model.pkl`
- `metadata/` - Training metadata and performance summary:
  - `sentiment_model_metadata.json`

### `src/`

Contains the distributed processing pipeline and analysis tools:

- `pipeline.py` - Core Spark-based sentiment analysis and term mining pipeline.
- `run_pipeline.py` - CLI wrapper for running the pipeline.
- `runner.py` - Full workflow orchestrator for pipeline, scalability, validation, and visualization.
- `validate.py` - Validates model predictions and tokenization.
- `visualize_results.py` - Creates dashboard visualizations from pipeline outputs.
- `scalability.py` - Benchmarks pipeline performance.
- `results/` - Generated output artifacts.
- `test_results/` - Optional experimental output directory.

## Model Training

### Purpose

The model training pipeline builds a sentiment classifier from multiple datasets and saves the resulting artifacts and train/test data.

### Data Sources

- `SetFit/sst5` — movie reviews
- `Yelp/yelp_review_full` — restaurant reviews
- `cardiffnlp/tweet_eval` — tweet sentiment

### Training Workflow

1. Load datasets from Hugging Face.
2. Standardize labels to three classes: `negative`, `neutral`, `positive`.
3. Sample balanced subsets for each domain.
4. Add `group` metadata and prefix text with the dataset category.
5. Split data into 80% train and 20% test sets.
6. Train a TF-IDF + Logistic Regression model.
7. Save model artifacts, metadata, and train/test CSV files.

### Saved Artifacts

- `model/models/tfidf_vectorizer.pkl`
- `model/models/sentiment_model.pkl`
- `model/metadata/sentiment_model_metadata.json`
- `data/train.csv`
- `data/test.csv`

## Pipeline Overview

### `src/pipeline.py`

This is the project’s core distributed processing engine. It:

- Reads input CSV data.
- Applies the saved TF-IDF vectorizer and classifier to text.
- Uses Spark `mapPartitions` for distributed inference.
- Tokenizes text and aggregates top terms by sentiment and group.
- Writes predictions, term reports, and runtime metadata.

### Outputs

- `predictions.csv`
- `group_term_sentiment_report.json`
- `pipeline_runtime.json`

## Workflow Orchestration

### `src/runner.py`

Runs the full workflow end-to-end:

- sentiment analysis pipeline
- scalability experiments
- validation
- visualization

### Recommended command

```bash
cd src
python runner.py --input-csv ../data/test.csv --output-dir results
```

## Validation and Visualization

### `src/validate.py`

Validates predictions by comparing them against the ground truth labels in `data/test.csv`.

- Computes classification metrics and confusion matrix.
- Validates tokenization and term frequency extraction.
- Saves results to JSON.

### `src/visualize_results.py`

Generates a dashboard with:

- runtime and performance charts
- confusion matrix visualization
- top sentiment terms by group
- sentiment distribution

### `src/scalability.py`

Benchmarks pipeline performance across:

- different document counts
- different chunk sizes
- fixed worker counts

## Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

### Java requirement

PySpark requires Java 17. On macOS:

```bash
brew install openjdk@17
echo 'export PATH="/opt/homebrew/opt/openjdk@17/bin:$PATH"' >> ~/.zshrc
echo 'export JAVA_HOME="/opt/homebrew/opt/openjdk@17"' >> ~/.zshrc
source ~/.zshrc
java -version
```

## Usage

### Train the model

```bash
cd model
python train_model.py
```

### Run the full workflow

```bash
cd src
python runner.py --input-csv ../data/test.csv --output-dir results
```

### Run just the pipeline

```bash
cd src
python run_pipeline.py --input-csv ../data/test.csv --output-dir results
```

### Validate predictions

```bash
cd src
python validate.py --input-csv ../data/test.csv --predictions-csv results/predictions.csv
```

### Generate the dashboard

```bash
cd src
python visualize_results.py --scalability-csv results/scalability_results.csv --validation-json results/validation_results.json --group-report-json results/group_term_sentiment_report.json --predictions-csv results/predictions.csv --output-png results/dashboard.png
```

### Run scalability tests

```bash
cd src
python scalability.py --input-csv ../data/test.csv --doc-counts "100,500,1000" --chunk-sizes "500,1000,2000"
```

## Notes

- Use `data/test.csv` for pipeline evaluation and validation.
- The pipeline supports inputs without a `group` column by defaulting missing values to `unknown`.
- The project is designed for reproducible training and scalable evaluation.
