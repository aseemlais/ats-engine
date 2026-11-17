FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including cmake and OpenBLAS for accelerated llama.cpp CPU inference
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install llama-cpp-python with OpenBLAS acceleration for lightweight CPU inference
RUN CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install --no-cache-dir llama-cpp-python

# Pre-download SentenceTransformer model during build for instant startup
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application source code
COPY . .

# Environment defaults for Hugging Face Spaces
ENV PORT=7860 \
    PYTHONUNBUFFERED=1 \
    USE_NEW_EXTRACTOR=false \
    GGUF_MODEL_PATH=/app/models/unsloth.Q4_K_M.gguf

# Hugging Face Spaces listens on port 7860
EXPOSE 7860

# Run a single uvicorn worker process to prevent memory thrashing with the background queue
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
