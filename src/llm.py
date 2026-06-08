from langchain_ollama import ChatOllama
from config import (
    MODEL_NAME,
    OLLAMA_BASE_URL,
    OLLAMA_API_KEY
)

llm = ChatOllama(
    model=MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    headers={
        "Authorization":
        f"Bearer {OLLAMA_API_KEY}"
    }
)


def invoke_llm(prompt: str):

    response = llm.invoke(prompt)

    return response.content