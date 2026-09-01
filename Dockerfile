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
    portaudio19-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

# Clone Fish Speech repository (using main branch for stability)
RUN git clone https://github.com/fishaudio/fish-speech.git /app/fish-speech && \
    cd /app/fish-speech && \
    git checkout main && \
    pip install --no-cache-dir -e .

# Create directory structure for presets and models
RUN mkdir -p /app/presets /app/models /app/models/fish-speech-1.5

# Pre-download Fish Speech 1.5 model weights
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('fishaudio/fish-speech-1.5', local_dir='/app/models/fish-speech-1.5', ignore_patterns=['*.git*'])"

# Pre-download preset voices
RUN echo "Downloading preset voice 1..." && \
    wget -q -O /app/presets/voice_1.wav https://files.catbox.moe/b1vfng.wav && \
    echo "Downloading preset voice 2..." && \
    wget -q -O /app/presets/voice_2.mp3 https://files.catbox.moe/i87vs7.mp3 && \
    echo "Downloading preset voice 3..." && \
    wget -q -O /app/presets/voice_3.mp3 https://files.catbox.moe/gr8o75.mp3

# Convert MP3 presets to WAV for consistency
RUN ffmpeg -i /app/presets/voice_2.mp3 -acodec pcm_s16le -ar 44100 /app/presets/voice_2.wav -y && \
    ffmpeg -i /app/presets/voice_3.mp3 -acodec pcm_s16le -ar 44100 /app/presets/voice_3.wav -y && \
    rm /app/presets/voice_2.mp3 /app/presets/voice_3.mp3

# Copy handler script
COPY handler.py /app/

# Health check to verify service readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import runpod; print('OK')" || exit 1

# Entrypoint
CMD ["python", "-u", "/app/handler.py"]
