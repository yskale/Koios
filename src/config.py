from langchain_community.embeddings import OllamaEmbeddings
from qdrant_client import QdrantClient
import os

OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434').rstrip('/')
QDRANT_URL = os.getenv('QDRANT_URL', 'http://localhost:53814').rstrip('/')
GEN_MODEL_NAME = os.getenv("GEN_MODEL_NAME", "llama3:latest")
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "llama3")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
STUDIES_JSON_FILE = os.getenv("STUDIES_JSON_FILE")

ollama_emb = OllamaEmbeddings(model=EMB_MODEL_NAME, base_url=OLLAMA_URL)

llm_url = f"{OLLAMA_URL}/api/generate"

QClient = QdrantClient(url=QDRANT_URL)
