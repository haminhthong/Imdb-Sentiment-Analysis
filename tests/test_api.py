"""Kiểm thử tự động cho FastAPI REST API trong api.py."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("httpx2")

from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "CineSentiment AI" in data["service"]
    assert "X-Request-ID" in response.headers


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "cinesentiment_requests_total" in response.text


def test_predict_validation_error():
    # Gửi request rỗng
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422  # Pydantic validation error


def test_predict_batch_validation_error():
    response = client.post("/predict/batch", json={"texts": []})
    assert response.status_code == 422
