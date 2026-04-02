FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY training/requirements.txt /app/training/requirements.txt
RUN pip install --no-cache-dir -r /app/training/requirements.txt

COPY training /app/training
COPY contracts /app/contracts

ENV PYTHONUNBUFFERED=1

CMD ["python", "training/train.py", "--config", "training/configs/baseline.yaml"]
