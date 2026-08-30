from pathlib import Path
from chromadb.utils import embedding_functions

# Project Paths
BASE_DIR = Path(__file__).parent.parent
DB_PATH = str(BASE_DIR / "email_db")
IDENTITY_PATH = str(BASE_DIR / "identity")
CREDENTIALS_FILE = str(IDENTITY_PATH / "credentials.json")
TOKEN_FILE = str(IDENTITY_PATH / "token.json")

# Chunking / Answering Model
CHUNKING_MODEL = "ollama/phi3.5:3.8b-mini-instruct-q4_K_M"
AVERAGE_CHUNK_SIZE = 1000
COLLECTION_NAME = "new_emails"

# How many emails to pull per ingest run.
# None = all messages in the folder(s) configured in ingest.py's fetch_emails()
MAX_RESULTS = None

# Embedding Configuration
EMBEDDING_MODEL_NAME = "nomic-embed-text:v1.5"
OLLAMA_EMBEDDING_URL = "http://localhost:11434/api/embeddings"

EMBEDDING_FUNCTION = embedding_functions.OllamaEmbeddingFunction(
    url=OLLAMA_EMBEDDING_URL,
    model_name=EMBEDDING_MODEL_NAME,
)

# Gmail Scopes
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]