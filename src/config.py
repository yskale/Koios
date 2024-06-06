from langchain_community.embeddings import OllamaEmbeddings
from qdrant_client import QdrantClient
import os

llama_port = os.getenv('LLAMA_PORT')
qdrant_port = os.getenv('QDRANT_PORT')

ollama_emb = OllamaEmbeddings(model="llama3", base_url=f"http://localhost:{llama_port}")


llm_url = f"http://localhost:{llama_port}/api/generate"


QClient = QdrantClient(url=f"http://localhost:{qdrant_port}")
#https://koios-qdrant.apps.renci.org/