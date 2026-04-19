from typing import List, Optional
from app.vector_store import vector_store
from app.schemas import DocumentResponse
from app.utils import get_logger

logger = get_logger(__name__)

def retrieve_documents(question: str, top_k: int = 3, topic_filter: Optional[str] = None) -> List[DocumentResponse]:
    """Retrieves relevant chunks from the vector store."""
    logger.info(f"Retrieving docs for question: '{question}', top_k: {top_k}, filter: {topic_filter}")
    
    results = vector_store.search(question, top_k, topic_filter)
    
    documents = []
    for doc, score in results:
        documents.append(
            DocumentResponse(
                file_name=doc.metadata.get("file_name", "unknown"),
                topic=doc.metadata.get("topic", "General"),
                chunk_id=doc.metadata.get("chunk_id", "unknown"),
                score=score,
                page_content=doc.page_content
            )
        )
        
    logger.info(f"Retrieved {len(documents)} context chunks.")
    return documents
