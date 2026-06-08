import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "gpt-oss:20b-cloud"

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL"
)

OLLAMA_API_KEY = os.getenv(
    "OLLAMA_API_KEY"
)