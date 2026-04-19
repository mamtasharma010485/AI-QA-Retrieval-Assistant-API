"""
test_ingest.py — Tests for the /ingest endpoint.

The /ingest endpoint reads .txt files from disk and loads them into ChromaDB.
These tests use mocking so no real files or API calls are needed.
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document


class TestIngestEndpoint:
    """Tests for POST /ingest"""

    def test_ingest_success(self, client):
        """
        Happy path: when documents exist and ChromaDB accepts them,
        the endpoint returns HTTP 200 with ingestion stats.
        """
        fake_docs = [
            Document(page_content="RAG stands for Retrieval-Augmented Generation.", metadata={"source": "rag_basics.txt"}),
            Document(page_content="AI Agents are autonomous systems.", metadata={"source": "ai_agents.txt"}),
        ]

        with patch("app.main.load_local_text_docs", return_value=fake_docs), \
             patch("app.main.ingest_documents", return_value={"num_documents": 2, "num_chunks": 4}):

            response = client.post("/ingest")
            assert response.status_code == 200

    def test_ingest_response_has_required_fields(self, client):
        """Response must contain 'message', 'num_documents', and 'num_chunks'."""
        fake_docs = [
            Document(page_content="Sample content.", metadata={"source": "sample.txt"}),
        ]

        with patch("app.main.load_local_text_docs", return_value=fake_docs), \
             patch("app.main.ingest_documents", return_value={"num_documents": 1, "num_chunks": 1}):

            response = client.post("/ingest")
            body = response.json()

            assert "message" in body
            assert "num_documents" in body
            assert "num_chunks" in body

    def test_ingest_returns_correct_counts(self, client):
        """num_documents and num_chunks in the response must match what was ingested."""
        fake_docs = [
            Document(page_content="Doc 1", metadata={"source": "a.txt"}),
            Document(page_content="Doc 2", metadata={"source": "b.txt"}),
        ]

        with patch("app.main.load_local_text_docs", return_value=fake_docs), \
             patch("app.main.ingest_documents", return_value={"num_documents": 2, "num_chunks": 5}):

            response = client.post("/ingest")
            body = response.json()

            assert body["num_documents"] == 2
            assert body["num_chunks"] == 5

    def test_ingest_no_documents_found_returns_404(self, client):
        """
        When 'data/docs/' is empty (no .txt files), the endpoint must return
        HTTP 404 — not a server crash.
        """
        with patch("app.main.load_local_text_docs", return_value=[]):
            response = client.post("/ingest")
            assert response.status_code == 404

    def test_ingest_404_has_meaningful_detail(self, client):
        """The 404 'detail' message should explain that no documents were found."""
        with patch("app.main.load_local_text_docs", return_value=[]):
            response = client.post("/ingest")
            body = response.json()

            assert "detail" in body
            assert len(body["detail"]) > 0

    def test_ingest_handles_unexpected_exception(self, client):
        """
        If the ingestion pipeline throws an unexpected error, the endpoint
        must return HTTP 500 (not crash the server).
        """
        with patch("app.main.load_local_text_docs", side_effect=RuntimeError("Disk read failed")):
            response = client.post("/ingest")
            assert response.status_code == 500
