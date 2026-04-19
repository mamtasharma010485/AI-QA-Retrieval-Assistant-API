from app.document_loader import load_documents, split_documents
from app.vector_store import vector_store
from app.config import settings
from app.utils import get_logger

logger = get_logger(__name__)

def ingest_data() -> dict:
    """Loads all documents from data dir, splits, and embeds them into Chroma DB."""
    
    logger.info(f"Loading documents from {settings.DATA_DIR}")
    docs = load_documents(settings.DATA_DIR)
    
    if not docs:
        logger.warning("No documents found to ingest.")
        return {"files_processed": 0, "chunks_created": 0}
        
    logger.info(f"Loaded {len(docs)} documents. Splitting into chunks...")
    chunks = split_documents(docs)
    
    logger.info(f"Created {len(chunks)} chunks. Adding to vector store...")
    vector_store.add_documents(chunks)
    
    logger.info("Ingestion complete.")
    return {
        "files_processed": len(docs),
        "chunks_created": len(chunks)
    }
