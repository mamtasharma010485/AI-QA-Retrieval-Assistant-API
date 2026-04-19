"""
test_schemas.py — Unit tests for the Pydantic schemas (app/schemas.py).

Schemas define what data is allowed IN and what is sent OUT of the API.
Testing them in isolation catches subtle bugs before they hit the endpoints.
"""

import pytest
from pydantic import ValidationError
from app.schemas import (
    AskRequest,
    AskResponse,
    DocumentSource,
    IngestResponse,
    HealthResponse,
)


class TestAskRequest:
    """Validates the incoming Q&A request schema."""

    def test_valid_question_is_accepted(self):
        """A plain string question must be parsed without error."""
        req = AskRequest(question="What is RAG?")
        assert req.question == "What is RAG?"

    def test_missing_question_raises_validation_error(self):
        """Omitting 'question' must raise a Pydantic ValidationError."""
        with pytest.raises(ValidationError):
            AskRequest()  # no question provided

    def test_question_field_is_string(self):
        """The parsed question must be a str type."""
        req = AskRequest(question="How does ChromaDB work?")
        assert isinstance(req.question, str)


class TestDocumentSource:
    """Validates the per-document source schema embedded in AskResponse."""

    def test_valid_document_source(self):
        """Both file_name and page_content must be accepted."""
        doc = DocumentSource(file_name="rag_basics.txt", page_content="Some chunk text...")
        assert doc.file_name == "rag_basics.txt"
        assert doc.page_content == "Some chunk text..."

    def test_missing_file_name_raises_error(self):
        """file_name is required — omitting it must raise ValidationError."""
        with pytest.raises(ValidationError):
            DocumentSource(page_content="Missing file name.")

    def test_missing_page_content_raises_error(self):
        """page_content is required — omitting it must raise ValidationError."""
        with pytest.raises(ValidationError):
            DocumentSource(file_name="some_file.txt")


class TestAskResponse:
    """Validates the outgoing Q&A response schema."""

    def test_valid_ask_response(self):
        """A complete AskResponse must be constructed without error."""
        resp = AskResponse(
            question="What is RAG?",
            answer="RAG is Retrieval-Augmented Generation.",
            documents=[
                DocumentSource(file_name="rag_basics.txt", page_content="RAG chunk...")
            ]
        )
        assert resp.question == "What is RAG?"
        assert resp.answer == "RAG is Retrieval-Augmented Generation."
        assert len(resp.documents) == 1

    def test_documents_can_be_empty_list(self):
        """An empty documents list is valid (no relevant docs found case)."""
        resp = AskResponse(
            question="Unknown question?",
            answer="The answer is not available in the provided documents.",
            documents=[]
        )
        assert resp.documents == []

    def test_missing_answer_raises_error(self):
        """'answer' is required; omitting it raises ValidationError."""
        with pytest.raises(ValidationError):
            AskResponse(question="Q?", documents=[])


class TestIngestResponse:
    """Validates the document ingestion response schema."""

    def test_valid_ingest_response(self):
        """A complete IngestResponse is constructed correctly."""
        resp = IngestResponse(message="Ingestion successful!", num_documents=2, num_chunks=5)
        assert resp.num_documents == 2
        assert resp.num_chunks == 5

    def test_num_documents_is_int(self):
        """num_documents must be an integer."""
        resp = IngestResponse(message="OK", num_documents=3, num_chunks=7)
        assert isinstance(resp.num_documents, int)

    def test_missing_num_chunks_raises_error(self):
        """'num_chunks' is required — omitting it raises ValidationError."""
        with pytest.raises(ValidationError):
            IngestResponse(message="OK", num_documents=2)


class TestHealthResponse:
    """Validates the health check response schema."""

    def test_valid_health_response(self):
        """A well-formed HealthResponse is accepted."""
        resp = HealthResponse(status="up", message="API is operational.")
        assert resp.status == "up"

    def test_missing_status_raises_error(self):
        """'status' is required — omitting it raises ValidationError."""
        with pytest.raises(ValidationError):
            HealthResponse(message="No status here.")

    def test_missing_message_raises_error(self):
        """'message' is required — omitting it raises ValidationError."""
        with pytest.raises(ValidationError):
            HealthResponse(status="up")
