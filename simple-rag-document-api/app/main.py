from fastapi import FastAPI, HTTPException
from app.schemas import HealthResponse, IngestResponse, AskRequest, AskResponse
from app.loader import load_local_text_docs
from app.rag import ingest_documents, ask_question
from typing import Dict, Any

# Create the FastAPI app instance
app = FastAPI(
    title="Simple RAG Document Q&A API",
    description="A beginner-friendly project for exploring Retrieval-Augmented Generation.",
    version="1.0.0"
)

# Root Path Configuration: where are our documents?
DOCS_DIRECTORY = "data/docs"

# Define the Health Check endpoint
# Use this to verify that the API is running correctly.
@app.get("/health", response_model=HealthResponse, tags=["Utility"])
async def health_check():
    """Returns a simple health response."""
    return {
        "status": "up",
        "message": "Simple RAG API is operational."
    }

# Define the Ingest endpoint
# Call this to load documents from your 'data/docs' folder into the system.
@app.post("/ingest", response_model=IngestResponse, tags=["Document Management"])
async def ingest_docs():
    """
    Scans the 'data/docs/' folder and stores its content in a local vector database.
    
    This must be called at least once before you can ask questions.
    """
    try:
        # Step 1: Load all text files from the disk
        docs = load_local_text_docs(DOCS_DIRECTORY)
        
        if not docs:
            raise HTTPException(status_code=404, detail="No documents found in data/docs folder.")

        # Step 2: Transfer documents into the search engine (ChromaDB)
        ingest_result = ingest_documents(docs)

        return {
            "message": "Ingestion successful!",
            "num_documents": ingest_result["num_documents"],
            "num_chunks": ingest_result["num_chunks"]
        }

    except Exception as e:
        # Simple error handling to catch and report what went wrong
        raise HTTPException(status_code=500, detail=str(e))

# Define the Ask endpoint
# This is the core functionality: processing user questions grounded in local data.
@app.post("/ask", response_model=AskResponse, tags=["Q&A"])
async def ask_docs(request: AskRequest):
    """
    Accepts a question and provides an answer grounded only in local documents.
    
    Example body:
    {
      "question": "What are AI Agents?"
    }
    """
    try:
        # Core retrieval and grounding logic
        result = await ask_question(request.question)

        return {
            "question": request.question,
            "answer": result["answer"],
            "documents": result["documents"]
        }

    except Exception as e:
        # Report issues (e.g., if OpenAI API is down or vector store is empty)
        raise HTTPException(status_code=500, detail=str(e))

# Simple entry point for running directly (python app/main.py)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
