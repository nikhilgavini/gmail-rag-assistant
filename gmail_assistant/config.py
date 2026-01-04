from pathlib import Path
from chromadb.utils import embedding_functions

# Project Paths
BASE_DIR = Path(__file__).parent.parent
DB_PATH = str(BASE_DIR / "email_db")
CREDENTIALS_FILE = str(BASE_DIR / "credentials.json")
TOKEN_FILE = str(BASE_DIR / "token.json")

# Model Configuration
# Note: Using 1b locally; higher models recommended if hardware allows or even frontier models if privacy not an issue
CHUNKING_MODEL = "ollama/llama3.2:1b" 
AVERAGE_CHUNK_SIZE = 1000
COLLECTION_NAME = "new_emails"
MAX_RESULTS = 1

# Local Embedding Configuration
EMBEDDING_FUNCTION = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Gmail Scopes
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]