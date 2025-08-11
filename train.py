import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from mlflow.models.signature import infer_signature

def main():
    # Configurar paths absolutos
    base_dir = Path(__file__).parent
    data_path = base_dir / "spam.csv"
    model_path = base_dir / "spam_model.joblib"
    training_data_path = base_dir / "training_data.csv"
    
    # Asegurar que el CSV existe
    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos en {data_path}")

    # Configuración de MLflow (usando servidor local)
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Spam Detection")
    
    # Cargar datos
    df = pd.read_csv(data_path)
    X = df["v2"]  # texto del mensaje
    y = df["v1"].map({'ham': 0, 'spam': 1})  # convertir a binario

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Crear pipeline (vectorizador + modelo)
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    
    # Entrenar
    pipeline.fit(X_train, y_train)

    # Evaluar
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Guardar modelo localmente
    joblib.dump(pipeline, model_path)
    print(f"Modelo guardado en {model_path}")
    
    # Guardar datos de entrenamiento como referencia
    df.to_csv(training_data_path, index=False)
    print(f"Datos de entrenamiento guardados en {training_data_path}")

    # Preparar firma y ejemplo de entrada
    signature = infer_signature(X_train, pipeline.predict(X_train))
    
    # Ejemplo de entrada (formato que acepta el pipeline)
    input_example = ["Free prize! Click now!"]

    with mlflow.start_run():
        # Loggear parámetros
        mlflow.log_params({
            "model_type": "LogisticRegression",
            "vectorizer": "CountVectorizer",
            "random_state": 42,
            "test_size": 0.2
        })

        # Loggear métricas
        report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_metrics({
            "accuracy": report["accuracy"],
            "precision_spam": report["1"]["precision"],
            "recall_spam": report["1"]["recall"],
            "f1_score_spam": report["1"]["f1-score"]
        })

        # Loggear el modelo
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="spam_classifier",
            signature=signature,
            input_example=input_example,
            registered_model_name="SpamClassifier"
        )

if __name__ == "__main__":
    # Iniciar servidor MLflow local si no está corriendo
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Asegúrate de tener el servidor MLflow corriendo:")
        print("mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns")
