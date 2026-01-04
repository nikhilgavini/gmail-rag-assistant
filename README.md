# Local Gmail RAG Assistant

A privacy-first Personal Knowledge Worker that transforms your Gmail inbox into a searchable knowledge base. 

This project implements a full Retrieval-Augmented Generation (RAG) pipeline using local LLMs (Ollama) to ensure your data never leaves your machine.

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

## ⚠️ Limitations & Future Work

* **Performance vs. Privacy Tradeoff:** 
  * Using a local 1b model (Ollama) prioritizes privacy and runs on consumer hardware but may lack the reasoning depth of "frontier" models (e.g., GPT-4o or Llama-3.1-70b).
* **Future Enhancements:** 
  * Implementation of **ThreadIDs** to maintain conversation context across multiple related emails.
  * Support for attachment parsing (PDF/Docx) within emails.



## Credits

* **Ed Donner:** Foundation and inspiration from the *AI Engineer Core Track (Week 5)*.

* **Jie Jenn:** Detailed Gmail API implementation guidance.
  * YouTube Link: https://www.youtube.com/watch?v=p7cn1n1kx3I

* **Google Gemini:** Assisted in architectural planning and framework optimization without providing code.