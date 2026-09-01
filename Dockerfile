FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y git ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# OpenBMB VoxCPM Model Download
RUN python -c "from voxcpm import VoxCPM; VoxCPM.from_pretrained('openbmb/VoxCPM2', load_denoiser=False)"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
