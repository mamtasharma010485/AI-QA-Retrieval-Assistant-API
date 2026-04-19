from typing import List, Optional
from langchain_openai import ChatOpenAI
from app.config import settings
from app.schemas import Message, DocumentResponse
from app.prompt_builder import build_messages
from app.utils import get_logger

logger = get_logger(__name__)

def get_chat_model():
    if settings.USE_LOCAL_MODELS:
        # Use local Ollama model
        try:
            from langchain_ollama import ChatOllama
            logger.info(f"Initializing local Ollama chat model: {settings.LOCAL_CHAT_MODEL}")
            return ChatOllama(
                model=settings.LOCAL_CHAT_MODEL,
                temperature=0.0,
                base_url=settings.OLLAMA_BASE_URL
            )
        except ImportError:
            logger.error("langchain_ollama not installed. Install with: pip install langchain-ollama")
            raise
    else:
        # Use OpenAI model
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set.")
        return ChatOpenAI(
            model=settings.CHAT_MODEL,
            temperature=0.0,
            openai_api_key=settings.OPENAI_API_KEY
        )

def generate_answer(question: str, chat_history: List[Message], documents: List[DocumentResponse]) -> str:
    """Generates an answer using the LLM and retrieved documents."""
    logger.info(f"Generating answer using {len(documents)} retrieved chunks.")

    chat_model = get_chat_model()
    messages = build_messages(question, chat_history, documents)

    response = chat_model.invoke(messages)

    return response.content
