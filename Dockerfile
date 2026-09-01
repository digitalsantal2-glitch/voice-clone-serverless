FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model Files Download
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='openbmb/VoxCPM2')"

# तीनों ऑडियो फाइलों को कंटेनर में पहले से सेव करें (ताकि रन-टाइम पर 0 सेकंड लगे)
RUN mkdir -p /app/presets && \
    curl -L -o /app/presets/long_kolhapuri.wav https://files.catbox.moe/b1vfng.wav && \
    curl -L -o /app/presets/competition_dialogue.mp3 https://files.catbox.moe/i87vs7.mp3 && \
    curl -L -o /app/presets/competition_voice.mp3 https://files.catbox.moe/gr8o75.mp3

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
