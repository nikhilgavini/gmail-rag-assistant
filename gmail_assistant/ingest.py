from gmail_utils import init_gmail_service, get_list_of_folders, get_email_messages, get_email_message_details
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from tqdm import tqdm
import config
import os
import time
from tenacity import retry, wait_exponential, stop_after_attempt

embed_wait = wait_exponential(multiplier=1, min=5, max=60)
load_dotenv(override=True)

###############################################################################
# VECTOR DB INFO
###############################################################################
chroma = PersistentClient(path=config.DB_PATH)
AVERAGE_CHUNK_SIZE = config.AVERAGE_CHUNK_SIZE
CHUNK_OVERLAP = config.CHUNK_OVERLAP


###############################################################################
# PYDANTIC CLASSES
###############################################################################
class Result(BaseModel):
    page_content: str
    metadata: dict


###############################################################################
## RAW DATA INGEST
###############################################################################
def get_since_query():
    if os.path.exists(config.LAST_INGEST_FILE):
        with open(config.LAST_INGEST_FILE) as f:
            epoch = f.read().strip()
        if epoch:
            # Small overlap buffer (1 hour) to absorb clock skew
            # Duplicate emails are still caught
            # by the existing_ids check below.
            return f"after:{max(int(epoch) - 3600, 0)}"
    return None  # first run ever: fetch everything


def save_ingest_checkpoint():
    with open(config.LAST_INGEST_FILE, "w") as f:
        f.write(str(int(time.time())))


def fetch_emails(service, max_results, since_query=None):
    emails = []
    seen_ids = set()

    # Loop through all of the folders
    for folder in get_list_of_folders(service):
        print(f"Looking in {folder} folder.")
        messages = get_email_messages(
            service, 
            max_results=max_results, 
            folder_name=folder, 
            query=since_query
        )
        # Extract the details and add them to the emails list of dicts
        for msg in messages:
            if msg['id'] in seen_ids:
                continue
            seen_ids.add(msg['id'])

            details = get_email_message_details(service, msg['id'])
            if details:
                details['metadata']['folder'] = folder
                emails.append({
                    'type': 'email',
                    'source': details['source'],
                    'text': details['text'],
                    'metadata': details['metadata']
                })
    
    return emails

###############################################################################
# CHUNKING
###############################################################################
def chunk_text(text, chunk_size=1000, overlap=200):
    text = text.strip()
    if not text:
        return []
 
    chunks = []
    start = 0
    length = len(text)
 
    while start < length:
        end = min(start + chunk_size, length)
        # If we're not at the end of the text, try to extend to the next
        # whitespace so we don't cut a word mid-token.
        if end < length:
            next_space = text.find(" ", end)
            if next_space != -1 and next_space - end < 50:
                end = next_space
 
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
 
        if end >= length:
            break
        start = max(end - overlap, start + 1)  # always make forward progress
 
    return chunks
 
 
def create_chunks(documents):
    header_lines = 3  # Subject / From / Date lines from _aggregate_email_text
 
    results = []
    for doc in tqdm(documents, desc="Processing Emails"):
        lines = doc["text"].split("\n", header_lines + 1)  # keep header intact
        header = "\n".join(lines[:header_lines])
        body = doc["text"]
 
        pieces = chunk_text(body, chunk_size=AVERAGE_CHUNK_SIZE, overlap=CHUNK_OVERLAP)
 
        for piece in pieces:
            # Avoid double-printing the header if this piece already starts
            # with it (true for the first chunk of a short email).
            page_content = piece if piece.startswith(header) else f"{header}\n\n{piece}"
            metadata = {
                "source": doc["source"],
                "type": doc["type"],
                "id": doc["metadata"]["id"],
            }
            results.append(Result(page_content=page_content, metadata=metadata))
 
    return results

###############################################################################
# EMBEDDING
###############################################################################
def get_existing_ids():
    collection = chroma.get_or_create_collection(
        name=config.COLLECTION_NAME, 
        embedding_function=config.EMBEDDING_FUNCTION
    )

    results = collection.get(include=["metadatas"])

    return {m["id"] for m in results["metadatas"] if "id" in m}


def create_embeddings(chunks, batch_size = 25):
    collection = chroma.get_or_create_collection(
        name=config.COLLECTION_NAME, 
        embedding_function=config.EMBEDDING_FUNCTION
    )

    total = len(chunks)
    for start in tqdm(range(0, total, batch_size), desc="Embedding batches"):
        batch = chunks[start:start + batch_size]
        texts = [chunk.page_content for chunk in batch]
        ids = [f"{chunk.metadata['id']}_{start + j}" for j, chunk in enumerate(batch)]
        metas = [chunk.metadata for chunk in batch]
        _add_batch(collection, ids, texts, metas)
 
    print(f"Vectorstore now has {collection.count()} documents (chunks)")

@retry(wait=embed_wait, stop=stop_after_attempt(3))
def _add_batch(collection, ids, texts, metas):
    collection.add(
        ids=ids, 
        documents=texts,
        metadatas=metas
    )

if __name__ == "__main__":
    client_secret_file = config.CREDENTIALS_FILE
    service = init_gmail_service(client_secret_file)

    since_query = get_since_query()
    if since_query:
        print(f"Incremental run: fetching messages matching '{since_query}'")
    else:
        print("First run: fetching all messages (this establishes the checkpoint).")

    documents = fetch_emails(service, config.MAX_RESULTS, since_query=since_query)
    print("Ingestion complete")
    print(f"{len(documents)} total emails found.")
    
    existing_ids = get_existing_ids()
    new_emails = [e for e in documents if e["metadata"]["id"] not in existing_ids]

    if not new_emails:
        print("No new emails to process.")
    else:
        print(f"Chunking {len(new_emails)} new emails.")
        chunks = create_chunks(new_emails)
        print("Proceeding with embeddings")
        create_embeddings(chunks)

    save_ingest_checkpoint()
