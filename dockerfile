# Imagen base ligera de Python
FROM python:3.10-slim

# Evitar que Python genere archivos .pyc y usar stdout para logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Crear directorio de la app
WORKDIR /app

# Copiar e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código y el modelo
COPY . .

# Exponer el puerto de FastAPI
EXPOSE 8000

# Comando para iniciar Uvicorn en producción
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
