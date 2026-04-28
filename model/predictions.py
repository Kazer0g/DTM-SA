#файл создан, чтобы показать как работает модель

import os
import joblib

from sklearn.pipeline import Pipeline


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "sentiment_model.pkl"
)
VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "models", 
    "tfidf_vectorizer.pkl"
)

# Load the individual components and reconstruct the pipeline
vectorizer = joblib.load(VECTORIZER_PATH)
classifier = joblib.load(MODEL_PATH)

model = Pipeline([
    ("tfidf", vectorizer),
    ("classifier", classifier)
])


examples = [
    {
        "text": "This movie was amazing",
        "category": "movies"
    },
    {
        "text": "The food was terrible",
        "category": "restaurants"
    },
    {
        "text": "It was okay, nothing special",
        "category": "tweets"
    }
]

model_inputs = [
    example["category"] + " " + example["text"]
    for example in examples
]

predictions = model.predict(model_inputs)

for example, prediction in zip(examples, predictions):
    print(example["text"], "->", prediction)
