import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import os

# Ensure local MLflow directory
os.makedirs("mlruns", exist_ok=True)
mlflow.set_tracking_uri("file:mlruns")

# Load dataset
df = pd.read_csv("spam.csv")  # adjust to your dataset path
X = df["text"]
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Save locally for Docker
joblib.dump((vectorizer, model), "spam_model.joblib")

# Log to MLflow with input example & signature
from mlflow.models.signature import infer_signature
import pandas as pd

input_example = pd.DataFrame({"text": ["Free prize! Click now!"]})
signature = infer_signature(X_train_vec, model.predict(X_train_vec))

with mlflow.start_run():
    mlflow.log_param("vectorizer", "CountVectorizer")
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("accuracy", model.score(X_test_vec, y_test))
    mlflow.sklearn.log_model(model, "model", signature=signature, input_example=input_example)
