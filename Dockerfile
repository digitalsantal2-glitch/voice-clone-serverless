FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Fish Speech 1.5 Download (Lightweight 1.5GB - कभी फेल नहीं होगा)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='fishaudio/fish-speech-1.5', local_dir='/app/checkpoints/fish-speech-1.5')"

# तीनों ऑडियो फाइल्स को प्री-डाउनलोड करें
RUN mkdir -p /app/presets && \
    curl -L -o /app/presets/long_kolhapuri.wav https://files.catbox.moe/b1vfng.wav && \
    curl -L -o /app/presets/competition_dialogue.mp3 https://files.catbox.moe/i87vs7.mp3 && \
    curl -L -o /app/presets/competition_voice.mp3 https://files.catbox.moe/gr8o75.mp3

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
