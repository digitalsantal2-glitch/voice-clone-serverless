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
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Fish Speech 1.5 का स्टेबल वर्जन इंस्टॉल करें
RUN git clone --branch v1.5.0 https://github.com/fishaudio/fish-speech.git /app/fish-speech && \
    cd /app/fish-speech && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir runpod soundfile requests

# Fish Speech 1.5 Model Download (1.5GB)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='fishaudio/fish-speech-1.5', local_dir='/app/checkpoints/fish-speech-1.5')"

# तीनों ऑडियो फाइल्स को प्री-डाउनलोड करें
RUN mkdir -p /app/presets && \
    curl -L -o /app/presets/long_kolhapuri.wav https://files.catbox.moe/b1vfng.wav && \
    curl -L -o /app/presets/competition_dialogue.mp3 https://files.catbox.moe/i87vs7.mp3 && \
    curl -L -o /app/presets/competition_voice.mp3 https://files.catbox.moe/gr8o75.mp3

COPY handler.py /app/handler.py

ENV PYTHONPATH="/app/fish-speech:$PYTHONPATH"

CMD ["python", "-u", "/app/handler.py"]
