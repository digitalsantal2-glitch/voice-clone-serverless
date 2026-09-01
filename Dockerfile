# Fish Audio - Speech Synthesis with RunPod Serverless Integration
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/app:$PATH" \
    FISH_SPEECH_HOME=/app/models

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    curl \
    wget \
    ca-certificates \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    sox \
    libsox-dev \
    pkg-config \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt /app/

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

# Create directory structure
RUN mkdir -p /app/presets /app/models /app/models/fish-speech-1.5

# Clone Fish Speech repository ONLY (skip model download in build)
RUN git clone --depth 1 https://github.com/fishaudio/fish-speech.git /app/fish-speech && \
    cd /app/fish-speech && \
    pip install --no-cache-dir -e . 2>&1 | head -50 || true

# Download ONLY the 3 preset voices (lightweight, ~50-100MB total)
RUN echo "Downloading preset voices..." && \
    wget --timeout=30 -q -O /app/presets/voice_1.wav https://files.catbox.moe/b1vfng.wav 2>&1 || echo "Warning: voice_1 download failed" && \
    wget --timeout=30 -q -O /app/presets/voice_2.mp3 https://files.catbox.moe/i87vs7.mp3 2>&1 || echo "Warning: voice_2 download failed" && \
    wget --timeout=30 -q -O /app/presets/voice_3.mp3 https://files.catbox.moe/gr8o75.mp3 2>&1 || echo "Warning: voice_3 download failed"

# Convert MP3 to WAV only if files exist
RUN bash -c 'if [ -f /app/presets/voice_2.mp3 ]; then ffmpeg -i /app/presets/voice_2.mp3 -acodec pcm_s16le -ar 44100 /app/presets/voice_2.wav -y 2>/dev/null && rm /app/presets/voice_2.mp3; fi' && \
    bash -c 'if [ -f /app/presets/voice_3.mp3 ]; then ffmpeg -i /app/presets/voice_3.mp3 -acodec pcm_s16le -ar 44100 /app/presets/voice_3.wav -y 2>/dev/null && rm /app/presets/voice_3.mp3; fi'

# Copy handler script
COPY handler.py /app/

# Create a startup script that downloads models on first run
RUN mkdir -p /app/scripts && \
    cat > /app/scripts/startup.sh << 'STARTUP_EOF'
#!/bin/bash
set -e

echo "[INFO] Checking if models need to be downloaded..."

if [ ! -f "/app/models/fish-speech-1.5/model.pth" ]; then
    echo "[INFO] Downloading Fish Speech 1.5 models..."
    python -c "from huggingface_hub import snapshot_download; snapshot_download('fishaudio/fish-speech-1.5', local_dir='/app/models/fish-speech-1.5', ignore_patterns=['*.git*', '*.md'])" || echo "[WARNING] Model download may retry at runtime"
fi

echo "[INFO] Starting handler..."
exec python -u /app/handler.py
STARTUP_EOF

RUN chmod +x /app/scripts/startup.sh

# Entrypoint
CMD ["/app/scripts/startup.sh"]
