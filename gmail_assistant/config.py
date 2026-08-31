import os
from pathlib import Path
from chromadb.utils import embedding_functions

# Project Paths
BASE_DIR = Path(__file__).parent.parent
DB_PATH = str(BASE_DIR / "email_db")
IDENTITY_PATH = str(BASE_DIR / "identity")
CREDENTIALS_FILE = str(BASE_DIR / "identity" / "credentials.json")
TOKEN_FILE = str(BASE_DIR / "identity" / "token.json")

# Chunking
AVERAGE_CHUNK_SIZE = 1000
CHUNK_OVERLAP = int(AVERAGE_CHUNK_SIZE * 0.2)
COLLECTION_NAME = "new_emails"

# How many emails to pull per ingest run.
# None = all messages in the folder(s) configured in ingest.py's fetch_emails()
MAX_RESULTS = 5

# Embedding Configuration
EMBEDDING_MODEL_NAME = "nomic-embed-text:v1.5"
OLLAMA_EMBEDDING_URL = "http://localhost:11434/api/embeddings"

EMBEDDING_FUNCTION = embedding_functions.OllamaEmbeddingFunction(
    url=OLLAMA_EMBEDDING_URL,
    model_name=EMBEDDING_MODEL_NAME,
    timeout=120 # So we have enough headroom
)

# Inference Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
print(f"[config] Ollama host: {OLLAMA_HOST}")
INFERENCE_MODEL = "phi3.5:3.8b-mini-instruct-q4_K_M"
RETRIEVAL_K = 10

# Gmail Scopes
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# Tracks unix epoch of last successful ingest
# Allows for only new messages each time
LAST_INGEST_FILE = str(BASE_DIR / "last_ingest_epoch.txt")