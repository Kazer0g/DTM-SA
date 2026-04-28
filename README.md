# DTM-SA: Distributed Text Mining & Sentiment Analysis

## Project Overview

This project implements a distributed text mining and sentiment analysis pipeline using PySpark and scikit-learn. The system processes large-scale text data to perform sentiment classification and extract meaningful term frequency patterns across different document groups.

## Project Structure

### Root Directory Files

- **`README.md`** - This file. Contains project overview, setup instructions, and usage guide.
- **`REPORT.md`** - Project report and documentation (currently empty).
- **`requirements.txt`** - Python dependencies required for the project.
- **`.gitignore`** - Git ignore rules for version control.
- **`.git/`** - Git repository metadata.
- **`.venv/`** - Python virtual environment (created during setup).

### `docs/` Directory - Documentation

Contains project documentation and guidelines:

- **`GUIDE.md`** - Developer guide outlining workflow, branching strategy, and best practices for contributing to the project.
- **`PROJECT-RULES.md`** - Project rules, workflow, and issue board management guidelines.

### `model/` Directory - Machine Learning Models

Contains all machine learning model training and prediction code:

#### Core Files:
- **`train_model.py`** - Main training script for the sentiment analysis model.
- **`predictions.py`** - Demonstration script for running predictions on trained models.
- **`train_data_for_ai.csv`** - Sample training data for reference.

#### Generated Directories (after training):
- **`models/`** - Trained model artifacts:
  - `tfidf_vectorizer.pkl` - TF-IDF vectorizer for text feature extraction
  - `sentiment_model.pkl` - Logistic Regression classifier for sentiment prediction

- **`metadata/`** - Training metadata and statistics:
  - `sentiment_model_metadata.json` - Model training details, performance metrics, and configuration

### `src/` Directory - Processing Pipeline

Contains the distributed processing pipeline and analysis tools:

#### Core Pipeline:
- **`pipeline.py`** - Main distributed processing engine using PySpark for sentiment analysis and term mining.
- **`run_pipeline.py`** - Command-line interface for executing the sentiment analysis pipeline.
- **`runner.py`**: Complete workflow orchestrator that runs pipeline, scalability testing, validation, and visualization in sequence.

#### Analysis Tools:
- **`validate.py`** - Validation and quality assurance for pipeline outputs.
- **`visualize_results.py`** - Creates comprehensive dashboard visualizations from pipeline results.
- **`scalability.py`** - Performance benchmarking and scalability testing.

#### Generated Directories (after pipeline execution):
- **`output/`** - Main output directory for pipeline results.
- **`test_output/`** - Test output directory for development and testing.
- **`__pycache__/`** - Python bytecode cache (auto-generated).

## Model Directory Details

### `train_model.py`
**Purpose**: Core training script for the sentiment analysis model.

**Functionality**:
- Loads training data from multiple public datasets via Hugging Face:
  - `SetFit/sst5` (movie reviews)
  - `Yelp/yelp_review_full` (restaurant reviews)
  - `cardiffnlp/tweet_eval` (tweet sentiment)
- Maps multi-class labels to unified 3-class format: `negative`, `neutral`, `positive`
- Samples up to 12,000 examples per dataset to balance training data
- Combines datasets with category prefixes for domain adaptation
- Trains a TF-IDF vectorizer + Logistic Regression pipeline
- Evaluates model performance with comprehensive metrics (accuracy, F1, precision, recall)
- Saves trained model artifacts directly in the model folder:
  - `models/tfidf_vectorizer.pkl` - TF-IDF vectorizer
  - `models/sentiment_model.pkl` - Logistic Regression classifier
  - `models/label_encoder.pkl` - Label encoder for string labels
  - `metadata/sentiment_model_metadata.json` - Training metadata and statistics

**Key Parameters**:
- TF-IDF: lowercase, English stopwords, max 50k features, unigrams+bigrams
- Logistic Regression: max_iter=1000, balanced class weights
- 80/20 train/test split with stratification

### `predictions.py`
**Purpose**: Demonstration script showing how to use the trained model for inference.

**Functionality**:
- Loads the trained model components and reconstructs the pipeline
- Provides example predictions on sample texts from different categories
- Shows the model input format (category + " " + text)

### `train_data_for_ai.csv`
**Purpose**: Sample training data file for reference.

**Structure**: Contains columns for `text`, `group` (category), and `sentiment` labels.

## Pipeline Directory Details

### `pipeline.py` - **Core Pipeline Engine**
**Purpose**: The main distributed processing pipeline that performs sentiment analysis and term frequency mining on large datasets.

**Key Functions**:
- **`tokenize()`**: Breaks text into words using regex pattern `[A-Za-z']+`
- **`load_artifacts()`**: Loads the trained model components (vectorizer, classifier)
- **`_predict_partition()`**: Processes data in distributed partitions using PySpark
- **`reduce_term_counts_spark()`**: Aggregates term frequencies by sentiment and group
- **`run_pipeline()`**: Main pipeline orchestrator

**Workflow**:
1. **Setup**: Creates PySpark session with configurable workers
2. **Data Loading**: Reads CSV input, selects text/group columns
3. **Distributed Prediction**: Uses `mapPartitions` to process data across Spark workers in batches
4. **Term Mining**: Tokenizes texts, counts term frequencies by sentiment/group
5. **Output**: Saves predictions CSV, term frequency JSON, and runtime metadata

**Key Features**:
- Memory-efficient batch processing (configurable chunk sizes)
- Fault-tolerant distributed execution
- Configurable worker threads for parallel processing

### `runner.py` - **Complete Workflow Orchestrator**
**Purpose**: All-in-one script that runs the entire DTM-SA workflow automatically.

**Functionality**:
- Executes the sentiment analysis pipeline
- Performs scalability benchmarking
- Validates results against ground truth
- Generates comprehensive visualizations
- Provides detailed progress reporting

**Usage**: `python runner.py --input-csv ../model/train_data_for_ai.csv --output-dir results`

**Parameters**:
- `--input-csv`: Path to input data file
- `--output-dir`: Where to save all results
- `--max-docs`: Limit documents processed (optional)
- `--scalability-doc-counts`: Document counts for benchmarking
- `--scalability-chunk-sizes`: Chunk sizes for benchmarking
- `--skip-scalability`: Skip performance testing
- `--skip-visualization`: Skip dashboard generation

**Outputs**: All pipeline, validation, scalability, and visualization results in one directory.

### `validate.py` - **Quality Assurance & Validation**
**Purpose**: Validates pipeline outputs and compares against ground truth.

**Functions**:
- **`validate_term_frequencies()`**: Checks tokenization consistency by comparing manual vs computed tokenization
- **`validate_sentiment()`**: Measures prediction accuracy against labeled data

**Outputs**:
- Classification reports and confusion matrices
- Term frequency validation metrics
- JSON file with validation results

**Usage**: `python validate.py --input-csv data.csv --predictions-csv results/predictions.csv`

### `visualize_results.py` - **Results Visualization**
**Purpose**: Creates comprehensive dashboard visualizations from pipeline outputs.

**Charts Generated**:
1. **Runtime vs Chunk Size**: Performance curves
2. **Speedup Analysis**: Efficiency improvements
3. **Confusion Matrix**: Prediction accuracy visualization
4. **Top Terms**: Most frequent positive/negative terms across groups
5. **Sentiment Distribution**: Overall prediction breakdown

**Output**: Single PNG dashboard combining all visualizations.

**Usage**: `python visualize_results.py --output-png dashboard.png`

### `scalability.py` - **Performance Benchmarking**
**Purpose**: Tests pipeline performance across different configurations.

**Functionality**:
- Runs pipeline with varying document counts and chunk sizes
- Measures execution time and calculates speedup ratios
- Tests scalability characteristics

**Parameters**:
- `--doc-counts`: Different dataset sizes to test (e.g., "1000,10000,100000")
- `--chunk-sizes`: Different batch sizes (e.g., "500,2000,5000,10000")

**Output**: CSV with scalability metrics for analysis.

**Usage**: `python scalability.py --input-csv data.csv --doc-counts "1000,10000" --chunk-sizes "500,2000"`

## Model Training Process

1. **Data Acquisition**:
   - Download datasets from Hugging Face Hub
   - Standardize label formats across datasets
   - Sample balanced subsets (12k per domain)

2. **Preprocessing**:
   - Combine domain prefix with text: `"movies " + review_text`
   - Remove null/empty texts
   - Stratified train/test split

3. **Model Architecture**:
   ```
   Input Text → TF-IDF Vectorization → Logistic Regression → Sentiment Prediction
   ```

4. **Training**:
   - Fit TF-IDF on training texts
   - Train logistic regression with balanced class weights
   - Evaluate on held-out test set

5. **Artifact Generation**:
   - Save individual model components for pipeline compatibility
   - Record training statistics and performance metrics

## Pipeline Execution Logic

The pipeline follows a MapReduce-inspired architecture:

### Map Phase (Distributed Prediction)
- **Input**: CSV rows with text and group columns
- **Processing**: Partition data across Spark workers
- **Model Application**: Load artifacts and predict sentiment in batches
- **Output**: (row_id, group, text, predicted_sentiment) tuples

### Reduce Phase (Term Aggregation)
- **Input**: Prediction results
- **Tokenization**: Extract meaningful terms from texts
- **Filtering**: Keep only positive/negative sentiment documents
- **Counting**: Aggregate term frequencies by group × sentiment
- **Ranking**: Sort and select top N terms per combination

### Output Generation
- **Predictions CSV**: Complete prediction results
- **Term Report JSON**: Hierarchical term frequency data
- **Runtime Metadata**: Performance and configuration info

## Dependencies

```
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
pyspark
datasets  # For Hugging Face dataset loading
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Java 17 (Required for PySpark)
```bash
# On macOS
brew install openjdk@17
echo 'export PATH="/opt/homebrew/opt/openjdk@17/bin:$PATH"' >> ~/.zshrc
echo 'export JAVA_HOME="/opt/homebrew/opt/openjdk@17"' >> ~/.zshrc
source ~/.zshrc

# Verify installation
java -version  # Should show Java 17
```

### 3. Train the Model
```bash
cd model
python train_model.py
```

### 4. Run Predictions (Optional)
```bash
cd model
python predictions.py
```

### 5. Run the Full Pipeline
```bash
cd src
python run_pipeline.py --input-csv ../path/to/data.csv --output-dir results
```

## Usage Examples

### Complete Workflow (Recommended)
```bash
cd src
python runner.py --input-csv ../model/train_data_for_ai.csv --output-dir results
```

### Individual Components
```bash
cd src
# Run just the pipeline
python run_pipeline.py --input-csv ../model/train_data_for_ai.csv --output-dir results

# Run validation
python validate.py --input-csv ../model/train_data_for_ai.csv --predictions-csv results/predictions.csv

# Run visualization
python visualize_results.py --output-png results/dashboard.png

# Run scalability testing
python scalability.py --input-csv ../model/train_data_for_ai.csv --doc-counts "100,500" --chunk-sizes "500,1000"
```

## Project Architecture

The project follows a modular architecture:

1. **Model Training** (`model/`): Independent ML model development
2. **Distributed Processing** (`src/`): Scalable data processing pipeline
3. **Documentation** (`docs/`): Development and project guidelines
4. **Results Analysis**: Validation, visualization, and performance testing

All components are designed to work together while remaining independently testable and maintainable.