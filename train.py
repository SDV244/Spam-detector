import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from mlflow.models.signature import infer_signature

# Force MLflow to log to a relative path (works in CI/CD and locally)
mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns')}")
mlflow.set_experiment("spam-detector")

# Example dataset load — replace with your own CSV or dataset
data = pd.read_csv("spam.csv", encoding="latin-1")[['v1', 'v2']]
data.columns = ['label', 'message']

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(
    data['message'],
    data['label'],
    test_size=0.2,
    random_state=42
)

# Flatten y to avoid Series flatten warnings
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Define pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

with mlflow.start_run():
    # Train model
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Metrics
    acc = metrics.accuracy_score(y_test, y_pred)
    report = metrics.classification_report(y_test, y_pred, output_dict=True)

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metrics({
        "precision_ham": report['ham']['precision'],
        "recall_ham": report['ham']['recall'],
        "f1_ham": report['ham']['f1-score'],
        "precision_spam": report['spam']['precision'],
        "recall_spam": report['spam']['recall'],
        "f1_spam": report['spam']['f1-score'],
    })

    # Save model locally
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/spam_detector.pkl")

    # Infer model signature
    input_example = pd.DataFrame({"message": ["Hello, this is a test message!"]})
    signature = infer_signature(input_example, pipeline.predict(input_example))

    # Log model with signature & example
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        input_example=input_example,
        signature=signature
    )

print(f"Accuracy: {acc}")
