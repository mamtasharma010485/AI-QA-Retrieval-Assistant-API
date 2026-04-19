"""
test_health.py — Tests for the /health endpoint.

This is the simplest test in the suite — it verifies the API is reachable
and returns the expected status payload. Great starting point for beginners!
"""

import pytest


class TestHealthEndpoint:
    """Group all /health tests together for clear reporting."""

    def test_health_returns_200(self, client):
        """The /health endpoint must respond with HTTP 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        """Response must contain both 'status' and 'message' keys."""
        response = client.get("/health")
        body = response.json()

        assert "status" in body, "Missing 'status' field in health response"
        assert "message" in body, "Missing 'message' field in health response"

    def test_health_status_value(self, client):
        """The 'status' field must be 'up' — not down, error, or anything else."""
        response = client.get("/health")
        assert response.json()["status"] == "up"

    def test_health_message_is_string(self, client):
        """The 'message' field must be a non-empty string."""
        response = client.get("/health")
        message = response.json()["message"]

        assert isinstance(message, str)
        assert len(message) > 0

    def test_health_content_type_is_json(self, client):
        """Response must be JSON (Content-Type: application/json)."""
        response = client.get("/health")
        assert "application/json" in response.headers["content-type"]
