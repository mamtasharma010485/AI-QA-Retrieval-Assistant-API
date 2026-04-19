from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from app.schemas import (
    HealthResponse, AskRequest, AskResponse, RetrieveRequest, 
    RetrieveResponse, IngestResponse, BaseResponse
)
from app.ingest_service import ingest_data
from app.retriever_service import retrieve_documents
from app.rag_service import generate_answer
from app.vector_store import vector_store
from app.utils import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="AI QA Retrieval Assistant API",
    description="A complete end-to-end RAG API Server built with FastAPI, Langchain and ChromaDB.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc), "status": "error"}
    )

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok", service="AI QA Retrieval Assistant API")

@app.post("/ingest", response_model=IngestResponse)
def ingest():
    try:
        result = ingest_data()
        return IngestResponse(
            status="success",
            message="Documents ingested successfully",
            files_processed=result["files_processed"],
            chunks_created=result["chunks_created"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    if request.top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be at least 1.")
        
    try:
        docs = retrieve_documents(
            question=request.question,
            top_k=request.top_k,
            topic_filter=request.topic_filter
        )
        
        answer = generate_answer(
            question=request.question,
            chat_history=request.chat_history,
            documents=docs
        )
        
        return AskResponse(
            question=request.question,
            answer=answer,
            documents=docs
        )
    except ValueError as e:
        # Catch value errors like missing API key
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error in ask: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(e)}")

@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: RetrieveRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    try:
        docs = retrieve_documents(
            question=request.question,
            top_k=request.top_k,
            topic_filter=request.topic_filter
        )
        
        return RetrieveResponse(
            question=request.question,
            retrieved_count=len(docs),
            documents=docs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset", response_model=BaseResponse)
def reset():
    try:
        vector_store.clear()
        return BaseResponse(status="success", message="Vector store cleared successfully")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
