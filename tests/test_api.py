from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    with client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

def test_prediction_endpoint():
    with client:
        payload = {
            "feature_numeric_1": 42.5,
            "feature_numeric_2": 1.75,
            "feature_categorical_1": "categoria_a"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert data["model_version"] == "v1.0.0"