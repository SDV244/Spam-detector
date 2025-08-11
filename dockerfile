# Use official Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Copy trained model from GitHub Actions artifact
COPY models/ ./models/

# Set environment variables
ENV MODEL_PATH=/app/models/spam_detector.pkl
ENV VECTORIZER_PATH=/app/models/tfidf_vectorizer.pkl
ENV PYTHONUNBUFFERED=1

# Command to run your inference script
CMD ["python", "inference.py"]
