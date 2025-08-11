import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Enable autologging for sklearn
mlflow.sklearn.autolog()

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2)

# Vectorize
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# MLflow experiment
mlflow.set_experiment("Spam Detection")

with mlflow.start_run():
    # Model
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train_vec, y_train)

    # Predictions
    y_pred = model.predict(X_test_vec)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))

    # Log custom metric
    mlflow.log_metric("accuracy", acc)

    # Save model locally for Docker
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/spam_detector.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

    # Log model to MLflow
    mlflow.sklearn.log_model(model, "model")
