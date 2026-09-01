# Fish Audio - Speech Synthesis with RunPod Serverless Integration
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git curl wget ca-certificates \
    ffmpeg libsndfile1 libsndfile1-dev sox libsox-dev pkg-config \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p /app/presets /app/models/fish-speech-1.5

# Clone Fish Speech (minimal)
RUN git clone --depth 1 https://github.com/fishaudio/fish-speech.git /app/fish-speech

# Try to install fish-speech (non-blocking)
RUN cd /app/fish-speech && pip install -e . 2>&1 || echo "Fish Speech install had issues"

# Download preset voices (non-blocking)
RUN wget --timeout=30 -q -O /app/presets/voice_1.wav https://files.catbox.moe/b1vfng.wav 2>&1 || true && \
    wget --timeout=30 -q -O /app/presets/voice_2.mp3 https://files.catbox.moe/i87vs7.mp3 2>&1 || true && \
    wget --timeout=30 -q -O /app/presets/voice_3.mp3 https://files.catbox.moe/gr8o75.mp3 2>&1 || true

# Convert MP3 to WAV
RUN if [ -f /app/presets/voice_2.mp3 ]; then ffmpeg -i /app/presets/voice_2.mp3 -acodec pcm_s16le -ar 44100 /app/presets/voice_2.wav -y 2>/dev/null; rm -f /app/presets/voice_2.mp3; fi

RUN if [ -f /app/presets/voice_3.mp3 ]; then ffmpeg -i /app/presets/voice_3.mp3 -acodec pcm_s16le -ar 44100 /app/presets/voice_3.wav -y 2>/dev/null; rm -f /app/presets/voice_3.mp3; fi

# Copy handler
COPY handler.py /app/handler.py

# Create startup script using printf instead of heredoc
RUN printf '#!/usr/bin/env python3\nimport os\nimport sys\n\nmodel_path = "/app/models/fish-speech-1.5/model.pth"\nif not os.path.exists(model_path):\n    print("[INFO] Downloading Fish Speech 1.5 models...")\n    try:\n        from huggingface_hub import snapshot_download\n        snapshot_download("fishaudio/fish-speech-1.5", local_dir="/app/models/fish-speech-1.5", ignore_patterns=["*.git*", "*.md"])\n        print("[INFO] Models downloaded")\n    except Exception as e:\n        print(f"[WARNING] Model download failed: {e}")\n\nprint("[INFO] Starting RunPod handler...")\nos.execvp("python", ["python", "-u", "/app/handler.py"])\n' > /app/start.py && chmod +x /app/start.py

# Entrypoint
CMD ["python", "/app/start.py"]
