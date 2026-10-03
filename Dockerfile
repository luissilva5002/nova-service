FROM python:3.11-slim

# --- Install system dependencies & C++ build tools ---
# Avoid forcing x86-only flags; the image can be built for amd64 and arm64.
# ffmpeg: audio format conversion (e.g. webm/opus from browser mic -> PCM)
# libsqlite3-dev: SQLite core memory store
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ffmpeg \
    libsqlite3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Upgrade pip and install core requirements ---
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Copy application source code ---
COPY . .

# --- Expose API and Web UI port ---
EXPOSE 8000

# --- Start FastAPI server via Uvicorn ---
CMD ["uvicorn", "nova.main:app", "--host", "0.0.0.0", "--port", "8000"]
