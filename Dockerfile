FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    TRANSFORMERS_CACHE=/models \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    DATA_ROOT=/data/rag-data \
    EMBEDDING_MODEL=/models/BAAI--bge-base-en-v1.5 \
    RERANKER_MODEL=/models/BAAI--bge-reranker-base \
    GENERATOR_MODEL=/models/Qwen--Qwen2.5-7B-Instruct

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-runtime.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 && \
    pip install -r requirements-runtime.txt

COPY harborlens_runtime ./harborlens_runtime
COPY harborlens_api.py .
COPY kaggle/input/rag-data /data/rag-data

RUN useradd --create-home --uid 10001 appuser && \
    mkdir -p /models && \
    chown -R appuser:appuser /app
USER 10001

EXPOSE 8000
CMD ["uvicorn", "harborlens_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
