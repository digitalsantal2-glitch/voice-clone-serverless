FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y git ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir voxcpm soundfile runpod huggingface_hub

# HuggingFace से Model फाइल्स डाउनलोड करें (बिना GPU एरर के)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='openbmb/VoxCPM2')"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
