FROM python:3.12-slim-bookworm

# Install system dependencies for OpenCV and other C-bindings
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements files first to leverage Docker cache
COPY backend/requirements.txt backend/requirements.txt

# Install standard requirements (now ONNX based, PyTorch removed)
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy the backend and model source code
COPY backend/ backend/
COPY model/ model/

# Expose Cloud Run's default port
EXPOSE 8080

# Run the Flask app using Gunicorn
CMD gunicorn --bind 0.0.0.0:${PORT:-8080} --chdir backend --timeout 120 --workers 1 main:app
