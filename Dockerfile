FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# 👇 यहाँ build-essential gcc g++ जोड़ दिया गया है (C Compiler)
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model Files Download
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='openbmb/VoxCPM2')"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
