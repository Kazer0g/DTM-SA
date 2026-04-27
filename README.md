# DTM-SA: Distributed Text Mining & Sentiment Analysis

## Model Directory

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

## Usage

### Training the Model
```bash
cd model
python train_model.py
```

### Running Predictions
```bash
cd model
python predictions.py
```