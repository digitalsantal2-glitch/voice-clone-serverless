FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y git ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model पहले ही डाउनलोड कर लें ताकि चालू होने में 0 सेकंड लगे
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='fishaudio/s2-pro', local_dir='/app/checkpoints/s2-pro')"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
