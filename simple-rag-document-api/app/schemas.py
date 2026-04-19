from pydantic import BaseModel, Field
from typing import List

# --- Request Models ---

class AskRequest(BaseModel):
    """
    Schema for the user's question input.
    """
    question: str = Field(..., description="The question you want to ask the documents.")

# --- Response Models ---

class DocumentSource(BaseModel):
    """
    Represents a specific piece of retrieved content used for the answer.
    """
    file_name: str
    page_content: str

class AskResponse(BaseModel):
    """
    The structured response sent back for a question.
    """
    question: str
    answer: str
    documents: List[DocumentSource]

class IngestResponse(BaseModel):
    """
    The output after processed documents are stored.
    """
    message: str
    num_documents: int
    num_chunks: int

class HealthResponse(BaseModel):
    """
    Simple status check schema.
    """
    status: str
    message: str
