import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from mlflow.models.signature import infer_signature

def main():
    # Set up MLflow - using local file storage
    mlflow_dir = os.path.abspath("mlruns")
    os.makedirs(mlflow_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")

    # Load dataset
    df = pd.read_csv("spam.csv")
    X = df["v2"]  # message text
    y = df["v1"].map({'ham': 0, 'spam': 1})  # convert to binary labels

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Vectorize
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    model = LogisticRegression(max_iter=1000)  # Added max_iter for convergence
    model.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = model.predict(X_test_vec)
    print(classification_report(y_test, y_pred))

    # Save locally for Docker
    joblib.dump((vectorizer, model), "spam_model.joblib")

    # Prepare MLflow logging
    signature = infer_signature(X_train_vec, model.predict(X_train_vec))
    
    # Create input example that matches the expected format
    example_text = ["Free prize! Click now!"]
    example_vectorized = vectorizer.transform(example_text)
    input_example = {"text": example_text}  # More intuitive input format

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "vectorizer": "CountVectorizer",
            "model": "LogisticRegression",
            "random_state": 42,
            "test_size": 0.2
        })

        # Log metrics
        mlflow.log_metrics({
            "accuracy": model.score(X_test_vec, y_test),
            "precision_spam": classification_report(y_test, y_pred, output_dict=True)['1']['precision'],
            "recall_spam": classification_report(y_test, y_pred, output_dict=True)['1']['recall']
        })

        # Log the vectorizer and model together as a pipeline
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', model)
        ])
        
        # Log model with custom Python environment
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="spam_classifier",
            signature=signature,
            input_example=input_example,
            registered_model_name="SpamClassifier",
            pyfunc_predict_fn="predict"  # Ensures proper prediction interface
        )

        # Log the training data as an artifact
        df.to_csv("training_data.csv", index=False)
        mlflow.log_artifact("training_data.csv")

if __name__ == "__main__":
    main()
