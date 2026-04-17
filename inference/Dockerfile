FROM python:3.11-slim

WORKDIR /app

COPY inference/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY inference /app/inference

EXPOSE 8000

CMD ["uvicorn", "inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
