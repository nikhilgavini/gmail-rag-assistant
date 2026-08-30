# Local Gmail RAG Assistant

A privacy-first Personal Knowledge Worker that transforms your Gmail inbox into a searchable knowledge base. 

This project implements a full Retrieval-Augmented Generation (RAG) pipeline using local LLMs (Ollama) to ensure your data never leaves your machine.
* Note that the user can swap to an open-source or frontier LLM to trade privacy for performance.

(Screenshots at the end)

## Overview

Inspired by **Ed Donner's AI Engineer Core Track**, this assistant connects to the Gmail API, processes unread or specific folder emails, and allows for natural language querying through a Gradio interface.

### Key Features

* **100% Local Processing:** Uses `llama3.2:1b` via Ollama for chunking, query rewriting, and answering, ensuring total privacy.
* **Advanced RAG Pipeline:** Implements query rewriting and result re-ranking to improve retrieval accuracy.
* **Vectorized Search:** Leverages **ChromaDB** with `all-MiniLM-L6-v2` embeddings for fast, semantic document retrieval.
* **Asynchronous Ingestion:** Efficiently fetches and processes email data into chunks.

### Exploration Python Notebook
* Follow along the exploration.ipynb file to view my thought process on gathering data from Gmail

## Architecture

The system operates in two main phases:

1. **Ingestion Phase (`ingest.py`):**
* Fetches emails via the **Gmail API**.
* Uses a local LLM to intelligently chunk emails based on semantic content rather than just character count.
* Vectorizes chunks and stores them in a local ChromaDB instance.


2. **Inference Phase (`answer.py` & `app.py`):**
* **Query Rewriting:** The user's question is optimized by the LLM for better vector search.
* **Hybrid Retrieval:** Performs multiple searches and merges results.
* **Re-ranking:** A secondary LLM pass ensures the most relevant context is provided to the final prompt.
* **Response Generation:** Generates a concise answer based strictly on the retrieved email context.

### Technical Deep Dive: Advanced RAG Pipeline

This assistant goes beyond basic "vector search + prompt." 

It uses a multi-stage pipeline to ensure the LLM receives the most relevant context possible:

#### 1. Query Rewriting

Before searching the database, the system uses an LLM to rewrite the user's question into a more descriptive search query. 
This bridges the gap between how people ask questions and how information is actually phrased in their emails.

#### 2. Hybrid Retrieval & Merging

To maximize coverage, the system performs two parallel searches:

* **Original Query Search**: Finds direct matches for the user's exact wording.
* **Rewritten Query Search**: Finds semantically related content that might use different keywords.
The results are then merged to create a comprehensive candidate pool.

#### 3. LLM-Based Re-ranking

Because vector search (ChromaDB in this case) can sometimes return "noisy" results, a secondary LLM pass acts as a re-ranker. 

It evaluates the candidate chunks against the original question and prioritizes only the most relevant snippets for the final answer.

#### 4. Robust Execution

The system implements **Exponential Backoff Retries** (via the `tenacity` library) for all LLM calls. 

This ensures that temporary rate limits or local hardware stutters don't break the application during long-running tasks like ingestion.

## Setup & Installation

### Prerequisites

* Python 3.10+
* [Ollama](https://ollama.com/) installed and running locally.
* Google Cloud Project with Gmail API enabled and `credentials.json` downloaded.

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/nikhilgavini/gmail-rag-assistant.git
cd gmail-rag-assistant

```

2. **Install dependencies:**
```bash
pip install -r requirements.txt

```

3. **Configure Environment:**
* Place your `credentials.json` in the root directory.
* Create a `.env` file for any optional remote API keys (Groq/OpenAI).
* Review `config.py` to ensure paths and model names match your local setup.


## Usage

1. **Ingest Data:**
Run the ingestion script to sync your emails to the local vector store.
```bash
python gmail_assistant/ingest.py

```

2. **Launch Assistant:**
Start the Gradio web interface.
```bash
python app.py

```

## Screenshots
<img width="1875" height="926" alt="Screenshot 2026-01-03 151213" src="https://github.com/user-attachments/assets/43c294e0-61ad-4b0b-b9ac-359bcd4bd8a9" />
<img width="1889" height="869" alt="Screenshot 2026-01-03 150722" src="https://github.com/user-attachments/assets/329039e7-9c61-49ad-b603-b41e082e7beb" />


## Limitations & Future Work

* **Performance vs. Privacy Tradeoff:** 
  * Using a local 1b model (Ollama) prioritizes privacy and runs on consumer hardware but may lack the reasoning depth of "frontier" models (e.g., GPT-4o or Llama-3.1-70b).
* **Future Enhancements:** 
  * Implementation of **ThreadIDs** to maintain conversation context across multiple related emails.
  * Support for attachment parsing (PDF/Docx) within emails.
  * Adding evaluation scripts to assess RAG metrics
    * Mean Reciprocal Rank (MRR)
    * Normalized Discounted Cumulative Gain (nDCG)
    * Recall@K
    * Precision@K



## Credits

* **Ed Donner:** Foundation and inspiration from the *AI Engineer Core Track (Week 5)*.

* **Jie Jenn:** Detailed Gmail API implementation guidance.
  * YouTube Link: https://www.youtube.com/watch?v=p7cn1n1kx3I

* **Google Gemini:** Assisted in architectural planning and framework optimization without providing code.
