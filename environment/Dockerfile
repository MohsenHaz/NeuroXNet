FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3.10 python3-pip git libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/neuroxnet-repro
COPY environment/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip==23.3.2 && \
    python3 -m pip install --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118 && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt
COPY . .
ENV PYTHONPATH=/workspace/neuroxnet-repro/src
CMD ["bash"]
