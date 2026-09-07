FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHECKPOINT_PATH=/app/artifacts/releases/v1.0.0/model.pt \
    RATE_LIMIT_PER_MINUTE=60

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sentiment ./sentiment
COPY api.py .
COPY artifacts ./artifacts

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
