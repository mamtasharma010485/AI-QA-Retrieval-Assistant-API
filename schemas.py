from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class Message(BaseModel):
    role: str
    content: str

class RetrieveRequest(BaseModel):
    question: str = Field(..., description="The user's question to retrieve context for")
    top_k: int = Field(default=3, description="Number of documents to retrieve")
    topic_filter: Optional[str] = Field(default=None, description="Optional topic to filter by")

class AskRequest(BaseModel):
    question: str = Field(..., description="The user's question")
    chat_history: List[Message] = Field(default=[], description="Previous conversation history")
    top_k: int = Field(default=3, description="Number of documents to retrieve")
    topic_filter: Optional[str] = Field(default=None, description="Optional topic to filter by")

class DocumentResponse(BaseModel):
    file_name: str
    topic: str
    chunk_id: str
    score: float
    page_content: str

class RetrieveResponse(BaseModel):
    question: str
    retrieved_count: int
    documents: List[DocumentResponse]

class AskResponse(BaseModel):
    question: str
    answer: str
    documents: List[DocumentResponse]

class HealthResponse(BaseModel):
    status: str
    service: str

class BaseResponse(BaseModel):
    status: str
    message: str

class IngestResponse(BaseResponse):
    files_processed: int
    chunks_created: int
