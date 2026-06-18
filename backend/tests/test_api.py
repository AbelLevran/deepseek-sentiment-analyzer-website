from fastapi.testclient import TestClient
import sys
import os

# Add backend directory to sys.path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_single():
    response = client.post("/predict", json={"text": "DeepSeek is amazing and I love it!"})
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "score" in data
    assert data["label"] in ["Positif", "Negatif", "Netral"]

def test_predict_batch():
    response = client.post("/predict-batch", json={"texts": ["Good app", "Very bad experience"]})
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 2
    assert data["results"][0]["label"] in ["Positif", "Negatif", "Netral"]

def test_dashboard_data():
    response = client.get("/dashboard-data")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "distribution" in data
