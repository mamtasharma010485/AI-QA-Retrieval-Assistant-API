import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "AI QA Retrieval Assistant API" in response.json()["service"]

def test_ask_empty_question():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 400

def test_retrieve_empty_question():
    response = client.post("/retrieve", json={"question": ""})
    assert response.status_code == 400
