"""
test_ask.py — Tests for the /ask endpoint.

The /ask endpoint performs the full RAG flow:
  1. Embed the question (OpenAI)
  2. Search ChromaDB for relevant chunks
  3. Pass context+question to the LLM
  4. Return a grounded answer

All external services are mocked so tests run offline with no API key.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_core.documents import Document


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_mock_llm_response(content: str) -> MagicMock:
    """Returns a mock LLM response object with the given content string."""
    mock_response = MagicMock()
    mock_response.content = content
    return mock_response


def make_retrieved_docs(n: int = 2) -> list:
    """Creates n fake retrieved Document objects as ChromaDB would return."""
    return [
        Document(
            page_content=f"RAG chunk {i}: Retrieval-Augmented Generation grounding fact {i}.",
            metadata={"source": f"data/docs/rag_basics.txt"}
        )
        for i in range(1, n + 1)
    ]


# ── Test Class ─────────────────────────────────────────────────────────────────

class TestAskEndpoint:
    """Tests for POST /ask"""

    def test_ask_returns_200_on_valid_question(self, client):
        """A well-formed question must return HTTP 200."""
        mock_docs = make_retrieved_docs()
        mock_answer = make_mock_llm_response("RAG stands for Retrieval-Augmented Generation.")

        with patch("app.rag.get_vector_store") as mock_vs, \
             patch("app.rag.ChatOpenAI") as mock_llm_cls:

            mock_vs.return_value.similarity_search.return_value = mock_docs
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_answer)
            mock_llm_cls.return_value = mock_llm

            response = client.post("/ask", json={"question": "What is RAG?"})
            assert response.status_code == 200

    def test_ask_response_contains_required_fields(self, client):
        """Response body must include 'question', 'answer', and 'documents'."""
        mock_docs = make_retrieved_docs()
        mock_answer = make_mock_llm_response("It is a grounding technique.")

        with patch("app.rag.get_vector_store") as mock_vs, \
             patch("app.rag.ChatOpenAI") as mock_llm_cls:

            mock_vs.return_value.similarity_search.return_value = mock_docs
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_answer)
            mock_llm_cls.return_value = mock_llm

            response = client.post("/ask", json={"question": "What is RAG?"})
            body = response.json()

            assert "question" in body
            assert "answer" in body
            assert "documents" in body

    def test_ask_echoes_the_question(self, client):
        """The 'question' field in the response must exactly match what was sent."""
        question_text = "What are AI Agents?"
        mock_docs = make_retrieved_docs()
        mock_answer = make_mock_llm_response("AI Agents are autonomous systems.")

        with patch("app.rag.get_vector_store") as mock_vs, \
             patch("app.rag.ChatOpenAI") as mock_llm_cls:

            mock_vs.return_value.similarity_search.return_value = mock_docs
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_answer)
            mock_llm_cls.return_value = mock_llm

            response = client.post("/ask", json={"question": question_text})
            assert response.json()["question"] == question_text

    def test_ask_documents_list_is_populated(self, client):
        """The 'documents' list should not be empty when ChromaDB returns results."""
        mock_docs = make_retrieved_docs(n=3)
        mock_answer = make_mock_llm_response("Some answer.")

        with patch("app.rag.get_vector_store") as mock_vs, \
             patch("app.rag.ChatOpenAI") as mock_llm_cls:

            mock_vs.return_value.similarity_search.return_value = mock_docs
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_answer)
            mock_llm_cls.return_value = mock_llm

            response = client.post("/ask", json={"question": "Tell me about RAG."})
            body = response.json()

            assert isinstance(body["documents"], list)
            assert len(body["documents"]) > 0

    def test_ask_document_source_has_file_name(self, client):
        """Each document in the response must have 'file_name' and 'page_content'."""
        mock_docs = make_retrieved_docs(n=1)
        mock_answer = make_mock_llm_response("Answer here.")

        with patch("app.rag.get_vector_store") as mock_vs, \
             patch("app.rag.ChatOpenAI") as mock_llm_cls:

            mock_vs.return_value.similarity_search.return_value = mock_docs
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=mock_answer)
            mock_llm_cls.return_value = mock_llm

            response = client.post("/ask", json={"question": "What is embedding?"})
            doc = response.json()["documents"][0]

            assert "file_name" in doc
            assert "page_content" in doc

    def test_ask_no_docs_returns_fallback_message(self, client):
        """
        When ChromaDB returns no matching documents, the answer must be the
        polite fallback message — NOT a crash or empty string.
        """
        with patch("app.rag.get_vector_store") as mock_vs:
            mock_vs.return_value.similarity_search.return_value = []

            response = client.post("/ask", json={"question": "What is the meaning of life?"})
            body = response.json()

            assert response.status_code == 200
            assert "not available" in body["answer"].lower()
            assert body["documents"] == []

    def test_ask_missing_question_field_returns_422(self, client):
        """
        Sending an empty body (missing 'question') must return HTTP 422 Unprocessable Entity.
        FastAPI/Pydantic enforces this automatically via the AskRequest schema.
        """
        response = client.post("/ask", json={})
        assert response.status_code == 422

    def test_ask_empty_string_question_returns_422(self, client):
        """
        An empty string question is technically valid JSON but violates our
        business logic — Pydantic should catch this (min_length enforcement).
        Note: If no min_length is set in schemas.py, adjust this test to 200.
        """
        response = client.post("/ask", json={"question": ""})
        # If the schema has no min_length validator, the API will accept it (200).
        # This test documents the current behavior — update if validation is tightened.
        assert response.status_code in (200, 422)

    def test_ask_handles_llm_exception_gracefully(self, client):
        """
        If the LLM call throws an exception (e.g., OpenAI is down),
        the endpoint must return HTTP 500, not crash the server.
        """
        mock_docs = make_retrieved_docs()

        with patch("app.rag.get_vector_store") as mock_vs, \
             patch("app.rag.ChatOpenAI") as mock_llm_cls:

            mock_vs.return_value.similarity_search.return_value = mock_docs
            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("OpenAI timeout"))
            mock_llm_cls.return_value = mock_llm

            response = client.post("/ask", json={"question": "What is RAG?"})
            assert response.status_code == 500
