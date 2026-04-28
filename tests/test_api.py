# tests/test_api.py

import pytest
from app import app


@pytest.fixture
def client():
    """
    Creates a test client for the Flask app
    """
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    """
    Ensures /health endpoint is alive
    """
    response = client.get("/health")
    assert response.status_code == 200


def test_predict_endpoint(client):
    """
    Ensures /predict endpoint works end-to-end
    """

    sample_input = {
        "feature1": 0,
        "feature2": 1,
        "feature3": 0
    }

    response = client.post("/predict", json=sample_input)

    assert response.status_code == 200
    assert response.json is not None