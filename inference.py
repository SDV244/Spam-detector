from flask import Flask, request, jsonify
import joblib
import os

MODEL_PATH = os.getenv("MODEL_PATH", "models/spam_detector.pkl")
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "models/tfidf_vectorizer.pkl")

# Load model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400
    
    text = data["text"]
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    label = "spam" if pred == 1 else "ham"
    
    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
