import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # OpenAI settings (optional, for fallback)
    OPENAI_API_KEY: str = ""

    # Local model settings
    USE_LOCAL_MODELS: bool = True
    LOCAL_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LOCAL_CHAT_MODEL: str = "llama3.2"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Legacy settings for backward compatibility
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    CHAT_MODEL: str = "gpt-4o-mini"

    # Other settings
    CHROMA_DB_DIR: str = "./chroma_db"
    DATA_DIR: str = "./data/docs"

    class Config:
        env_file = ".env"

settings = Settings()
