import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

class Config:
    """
    Configuration manager for the application.
    Handles environment variables with clear error feedback.
    """
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    @classmethod
    def validate(cls):
        """Validates that all required configuration is present."""
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "CRITICAL ERROR: OPENAI_API_KEY is missing from environment or .env file. "
                "Please add your API key to continue."
            )

# Run validation on import to catch issues early
Config.validate()
