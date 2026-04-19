import os
import shutil
from typing import List, Optional, Tuple
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from app.config import settings
from app.utils import get_logger, ensure_dir
import traceback

logger = get_logger(__name__)

# ChromaDB import - try langchain_chroma first (newer), fall back to langchain_community
try:
    from langchain_chroma import Chroma
    logger.info("Using langchain_chroma for Chroma vector store.")
except ImportError:
    from langchain_community.vectorstores import Chroma
    logger.info("Using langchain_community for Chroma vector store.")


class VectorStore:
    def __init__(self):
        self.persist_directory = settings.CHROMA_DB_DIR
        ensure_dir(self.persist_directory)
        self.embeddings = None
        self.db = None

    def _ensure_initialized(self):
        if not self.embeddings:
            if settings.USE_LOCAL_MODELS:
                # Use local HuggingFace embeddings
                try:
                    from langchain_huggingface import HuggingFaceEmbeddings
                    logger.info(f"Initializing local HuggingFace embeddings with model: {settings.LOCAL_EMBEDDING_MODEL}")
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name=settings.LOCAL_EMBEDDING_MODEL
                    )
                except ImportError:
                    logger.error("langchain_huggingface not installed. Install with: pip install langchain-huggingface")
                    raise
            else:
                # Use OpenAI embeddings
                if not settings.OPENAI_API_KEY:
                    raise ValueError("OPENAI_API_KEY is not set in your .env file.")
                logger.info(f"Initializing OpenAI embeddings with model: {settings.EMBEDDING_MODEL}")
                self.embeddings = OpenAIEmbeddings(
                    model=settings.EMBEDDING_MODEL,
                    openai_api_key=settings.OPENAI_API_KEY
                )

        if not self.db:
            logger.info(f"Initializing Chroma DB at: {self.persist_directory}")
            try:
                self.db = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name="rag_knowledge_base"
                )
                logger.info("Chroma DB initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Chroma DB: {e}\n{traceback.format_exc()}")
                raise

    def add_documents(self, documents: List[Document]) -> int:
        self._ensure_initialized()

        if not documents:
            return 0

        try:
            logger.info(f"Adding {len(documents)} chunks to Chroma DB...")
            self.db.add_documents(documents)
            logger.info(f"Successfully added {len(documents)} chunks to vector store.")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}\n{traceback.format_exc()}")
            raise
        return len(documents)

    def search(self, query: str, top_k: int = 3, topic_filter: Optional[str] = None) -> List[Tuple[Document, float]]:
        self._ensure_initialized()

        filter_dict = None
        if topic_filter:
            filter_dict = {"topic": topic_filter}

        try:
            results = self.db.similarity_search_with_relevance_scores(
                query=query,
                k=top_k,
                filter=filter_dict
            )
        except Exception as e:
            logger.error(f"Error during similarity search: {e}\n{traceback.format_exc()}")
            raise
        return results

    def clear(self):
        self.db = None
        self.embeddings = None
        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
                logger.info(f"Cleared chroma_db directory: {self.persist_directory}")
            except Exception as e:
                logger.error(f"Failed to clear chroma db: {e}")
        ensure_dir(self.persist_directory)
        logger.info("Vector store cleared and reset.")


vector_store = VectorStore()
