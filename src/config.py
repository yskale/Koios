from langchain_community.embeddings import OllamaEmbeddings
from qdrant_client import QdrantClient, async_qdrant_client
import os
from pathlib import Path
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434').rstrip('/')
QDRANT_URL = os.getenv('QDRANT_URL', 'http://localhost:6333').rstrip('/')
GEN_MODEL_NAME = os.getenv("GEN_MODEL_NAME", "llama3:latest")
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "llama3")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "test_collection2")
STUDIES_JSON_FILE = os.getenv("STUDIES_JSON_FILE", Path(os.path.dirname(__file__), '..',  'data', '99_select_studies.json'))
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", Path(os.path.dirname(__file__), '..' ,  'koios.log'))
LANGFUSE_ENABLED = os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "")

ollama_emb = OllamaEmbeddings(model=EMB_MODEL_NAME, base_url=OLLAMA_URL)

llm_url = f"{OLLAMA_URL}/api/generate"

QClient = QdrantClient(url=QDRANT_URL)
AQClient = async_qdrant_client.AsyncQdrantClient(url=QDRANT_URL)


def configure_langfuse(runnable):
    if LANGFUSE_ENABLED:
        from langchain_core.runnables.config import RunnableConfig
        from langfuse.callback import CallbackHandler

        langfuse_handler = CallbackHandler(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
        langfuse_handler.auth_check()
        runnable_config = RunnableConfig(callbacks=[langfuse_handler])
        return runnable.with_config(runnable_config)
    return runnable