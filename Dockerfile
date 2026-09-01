FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

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
    && rm -rf /var/lib/apt/lists/*

# Fish Speech ओरिजिनल कोड इंस्टॉल करें
RUN git clone --depth 1 https://github.com/fishaudio/fish-speech.git /app/fish-speech && \
    cd /app/fish-speech && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir runpod soundfile requests huggingface_hub

# कल वाला S2-Pro AI Model Download
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='fishaudio/s2-pro', local_dir='/app/checkpoints/s2-pro')"

# तीनों DJ ऑडियो फाइल्स डाउनलोड करें
RUN mkdir -p /app/presets && \
    curl -L -o /app/presets/long_kolhapuri.wav https://files.catbox.moe/b1vfng.wav && \
    curl -L -o /app/presets/competition_dialogue.mp3 https://files.catbox.moe/i87vs7.mp3 && \
    curl -L -o /app/presets/competition_voice.mp3 https://files.catbox.moe/gr8o75.mp3

COPY handler.py /app/handler.py

ENV PYTHONPATH="/app/fish-speech:$PYTHONPATH"

CMD ["python", "-u", "/app/handler.py"]
