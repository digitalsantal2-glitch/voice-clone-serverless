FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y git ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# S2-Pro Model Download
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='fishaudio/s2-pro', local_dir='/app/checkpoints/s2-pro')"

COPY handler.py /app/handler.py
# आपकी प्रीसेट ऑडियो फाइलें कॉपी करने के लिए (अगर हों)
RUN mkdir -p /app/presets

CMD ["python", "-u", "handler.py"]
