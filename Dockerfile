# ── Volarix — HF Spaces Dockerfile ──────────────────────────────────────────
# HF Spaces requires the app to listen on port 7860.
# The FROM image is slim Python to keep the layer cache fast.

FROM python:3.10-slim

# System deps (needed by some torch / scipy wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements 
COPY requirements.deploy.txt .

# Install Python deps
# --no-cache-dir keeps the image smaller
RUN pip install --no-cache-dir -r requirements.deploy.txt

# Copy the full project (src/ modules + api/ + data/ etc.)
# .dockerignore (see companion file) excludes .env, __pycache__, *.pt etc.
COPY . .

# HF Spaces mandatory: listen on 7860
EXPOSE 7860

# Uvicorn:
#   --host 0.0.0.0   — accept connections from outside the container
#   --port 7860      — HF Spaces requirement
#   --workers 1      — single worker; model is loaded in memory once
#   --timeout-keep-alive 75 — keeps long bulletin calls alive
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--timeout-keep-alive", "75"]
