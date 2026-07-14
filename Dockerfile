# syntax=docker/dockerfile:1
#
# Agentic EDA Pipeline — Streamlit app container.
# Ollama runs as a separate service (see docker-compose.yml), so this image
# only needs the Python app and its dependencies.

FROM python:3.11-slim

# System deps: build tools for scientific wheels + fonts for matplotlib/fpdf.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        fonts-dejavu-core \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    # Default: talk to the Ollama sidecar by service name.
    OLLAMA_HOST=http://ollama:11434

WORKDIR /app

# Install Python deps first for better layer caching.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the application code.
COPY . .

# Persisted at runtime via volumes (data/, outputs/).
RUN mkdir -p data/uploads data/embedchain outputs

EXPOSE 8501

# Basic container healthcheck against Streamlit's health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py"]
