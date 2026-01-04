from gmail_utils import init_gmail_service, get_list_of_folders, get_email_messages, get_email_message_details
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from tqdm import tqdm
from litellm import completion
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from tenacity import retry, wait_exponential, stop_after_attempt
import config
import time

load_dotenv(override=True)
###############################################################################
# MODELS (Chunking and Embedding)
###############################################################################
## Chunking Model
MODEL = config.CHUNKING_MODEL

## Local Embedding Model
local_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

###############################################################################
# VECTOR DB INFO
###############################################################################
chroma = PersistentClient(path=config.DB_PATH)
AVERAGE_CHUNK_SIZE = config.AVERAGE_CHUNK_SIZE

###############################################################################
# MULTIPROCESSING AND RETRIES
###############################################################################
WORKERS = 1
wait = wait_exponential(multiplier=1, min=4, max=60)

###############################################################################
# PYDANTIC CLASSES
###############################################################################
class Result(BaseModel):
    page_content: str
    metadata: dict


class Chunk(BaseModel):
    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query",
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, exactly as is, not changed in any way"
    )

    def as_result(self, document):
        metadata = {
            "source": document["source"], 
            "type": document["type"],
            "id": document['metadata']['id']
        }
        return Result(
            page_content=self.headline + "\n\n" + self.summary + "\n\n" + self.original_text,
            metadata=metadata,
        )


class Chunks(BaseModel):
    chunks: list[Chunk]


###############################################################################
## RAW DATA INGEST
###############################################################################
def fetch_emails(service, max_results):
    emails = []
    # Loop through all of the folders
    #for folder in get_list_of_folders(service):
    for folder in ['SPAM']:
        print(f"Looking in {folder} folder.")
        messages = get_email_messages(service, max_results=max_results, folder_name=folder)
        # Extract the details and add them to the emails list of dicts
        for msg in messages:
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
def make_prompt(document):
    how_many = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1
    return f"""
You take a document and you split the document into overlapping chunks for a KnowledgeBase.

The document is from the email account of the user.
The document is of type: {document["type"]}
The document is of subject: {document["source"]}

A chatbot will use these chunks to answer questions.
You should divide up the document as you see fit, being sure that the entire document is returned across the chunks - don't leave anything out.
This document should probably be split into at least {how_many} chunks, but you can have more or less as appropriate, ensuring that there are individual chunks to answer specific questions.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

For each chunk, you should provide a headline, a summary, and the original text of the chunk.
Together your chunks should represent the entire document with overlap.

Here is the document:

{document["text"]}

Respond with the chunks.
"""


def make_messages(document):
    return [
        {"role": "user", "content": make_prompt(document)},
    ]


@retry(wait=wait, stop=stop_after_attempt(2))
def process_document(document):
    # If running into issues with ollama, uncommment this so you can truncate to 2000 tokens
    #doc_to_process = document.copy()
    #doc_to_process["text"] = document["text"][:2000]    # Truncate because you can usually identify phishing attempts in the first 1500 words
    
    messages = make_messages(document)
    response = completion(
        model=MODEL, 
        messages=messages, 
        response_format=Chunks,
        timeout=None
    )

    reply = response.choices[0].message.content

    try:
        doc_as_chunks = Chunks.model_validate_json(reply).chunks
        return [chunk.as_result(document) for chunk in doc_as_chunks]
    except Exception as e:
        print(f"FAILED JSON: {reply[:100]}...{reply[-100:]}")
        raise e


def create_chunks(documents):
    chunks = []
    for doc in tqdm(documents, desc="Processing Emails"):
        try:
            result = process_document(doc)
            chunks.extend(result)
            
            # Add a 10-second 'Cool Down' to avoid RPM limits
            # Groq Free tier usually allows ~3-14 requests per minute
            time.sleep(10) 
            
        except Exception as e:
            if "rate_limit_exceeded" in str(e).lower():
                print("Rate limit hit, sleeping for 60s...")
                time.sleep(60)
                # Retry once
                result = process_document(doc)
                chunks.extend(result)
            else:
                print(f"Skipping email due to error: {e}")
    return chunks

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


def create_embeddings(chunks):
    collection = chroma.get_or_create_collection(
        name=config.COLLECTION_NAME, 
        embedding_function=config.EMBEDDING_FUNCTION
    )

    texts = [chunk.page_content for chunk in chunks]
    ids = [f"{chunk.metadata['id']}_{i}" for i, chunk in enumerate(chunks)]
    metas = [chunk.metadata for chunk in chunks]

    collection.add(ids=ids, documents=texts, metadatas=metas)
    print(f"Vectorstore created with {collection.count()} documents (chunks)")


if __name__ == "__main__":
    client_secret_file = config.CREDENTIALS_FILE
    service = init_gmail_service(client_secret_file)

    documents = fetch_emails(service, config.MAX_RESULTS)
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