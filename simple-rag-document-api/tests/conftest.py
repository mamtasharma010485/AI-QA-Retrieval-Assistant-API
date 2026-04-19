"""
conftest.py — Shared test fixtures for the Simple RAG API test suite.

Fixtures defined here are automatically available to ALL test files
without needing to import them explicitly.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# ── Patch OpenAI key before importing the app ─────────────────────────────────
# config.py raises ValueError if OPENAI_API_KEY is missing.
# We set a fake key here BEFORE the app is imported so validation passes.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake-key-for-testing")


@pytest.fixture(scope="session")
def client():
    """
    Creates a FastAPI TestClient that can be reused across the whole test session.

    scope="session" means it is created once and shared — avoids startup
    overhead for every individual test.
    """
    # Patch config validation so no real API key is needed
    with patch("app.config.Config.validate"):
        from app.main import app
        with TestClient(app) as c:
            yield c


@pytest.fixture
def mock_openai_embeddings():
    """Patches OpenAIEmbeddings so no real API call is made during tests."""
    with patch("app.rag.OpenAIEmbeddings") as mock_embed:
        mock_embed.return_value = MagicMock()
        yield mock_embed


@pytest.fixture
def mock_chroma():
    """Patches the ChromaDB vector store with a safe, in-memory fake."""
    with patch("app.rag.Chroma") as mock_chroma_cls:
        mock_store = MagicMock()
        mock_chroma_cls.return_value = mock_store
        yield mock_store


@pytest.fixture
def mock_chat_openai():
    """Patches ChatOpenAI so no real LLM call is triggered during tests."""
    with patch("app.rag.ChatOpenAI") as mock_llm_cls:
        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm
        yield mock_llm
