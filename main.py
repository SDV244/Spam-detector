from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# 1️⃣ Definir el esquema de entrada
class PredictRequest(BaseModel):
    text: str

# 2️⃣ Crear instancia de FastAPI
app = FastAPI(title="Spam Detector API", version="1.0")

# 3️⃣ Cargar modelo y vectorizador una sola vez
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# 4️⃣ Endpoint de predicción
@app.post("/predict")
def predict(request: PredictRequest):
    # Transformar texto
    text_vec = vectorizer.transform([request.text])

    # Realizar predicción
    prediction = model.predict(text_vec)[0]

    # Preparar respuesta
    return {
        "prediction": int(prediction),
        "label": "spam" if prediction == 1 else "ham"
    }
