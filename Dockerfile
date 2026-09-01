FROM fishaudio/fish-speech:latest-server-cuda

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN pip install --no-cache-dir runpod soundfile requests

# Model Download (100% Matches Container)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='fishaudio/s2-pro', local_dir='/app/checkpoints/s2-pro')"

# 3 Preset Audios Download
RUN mkdir -p /app/presets && \
    curl -L -o /app/presets/long_kolhapuri.wav https://files.catbox.moe/b1vfng.wav && \
    curl -L -o /app/presets/competition_dialogue.mp3 https://files.catbox.moe/i87vs7.mp3 && \
    curl -L -o /app/presets/competition_voice.mp3 https://files.catbox.moe/gr8o75.mp3

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
