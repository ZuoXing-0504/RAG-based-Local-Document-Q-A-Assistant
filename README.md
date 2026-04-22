# RAG-based Local Document Q&A Assistant

A lightweight, fully local Retrieval-Augmented Generation (RAG) project built for internship portfolios, course projects, and AI/LLM interview demos.

This application allows users to upload PDF and TXT files, clean and split the documents, build a persistent FAISS vector index, and ask questions through a Streamlit web interface. The whole pipeline runs on CPU and does not require model training or a GPU.

## Why This Project Is Resume-Friendly

- Built an end-to-end local RAG pipeline from document ingestion to answer generation
- Implemented document cleaning, chunking, deduplication, vector indexing, and retrieval
- Used open-source embedding models and FAISS for efficient local semantic search
- Designed a modular Python codebase that is easy to explain and extend in interviews
- Delivered a usable Streamlit frontend instead of only backend scripts

## Features

- Batch upload for `PDF` and `TXT` files
- PDF parsing with `PyPDF2`
- Text cleaning and duplicate-line removal
- Recursive chunk splitting with overlap
- Chunk-level deduplication with hashing
- Persistent local FAISS vector store
- Semantic retrieval for user questions
- Streamlit-based interactive Q&A interface
- CPU-only local deployment

## Tech Stack

- Python 3.9+
- Streamlit
- LangChain
- LangChain Community
- LangChain Text Splitters
- FAISS
- PyPDF2
- Sentence Transformers

## System Workflow

1. Upload PDF or TXT documents from the Streamlit sidebar
2. Parse raw text from files and normalize the content
3. Split documents into overlapping chunks and remove duplicates
4. Convert chunks into embeddings with an open-source embedding model
5. Store embeddings in a local FAISS index on disk
6. Retrieve top relevant chunks for each user query
7. Synthesize a concise answer and show supporting sources

## Project Structure

```text
RAG-based Local Document Q&A Assistant/
|-- app.py
|-- requirements.txt
|-- README.md
|-- LICENSE
|-- data/
|   |-- uploads/
|   `-- vector_store/
`-- rag_local_qa/
    |-- __init__.py
    |-- config.py
    |-- document_processor.py
    |-- qa_engine.py
    `-- vector_store.py
```

## Core Modules

### `app.py`

The Streamlit entrypoint. It provides:

- document upload
- knowledge base rebuild
- chat interaction
- source and chunk inspection

### `rag_local_qa/document_processor.py`

Handles:

- PDF and TXT loading
- text cleaning
- line deduplication
- chunk splitting
- chunk-level deduplication

### `rag_local_qa/vector_store.py`

Responsible for:

- embedding model loading
- FAISS index creation
- local persistence
- similarity search

### `rag_local_qa/qa_engine.py`

Implements:

- query retrieval
- keyword-aware sentence selection
- answer synthesis
- source formatting for the UI

## Quick Start

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

### 4. Use the application

1. Open the Streamlit page in your browser
2. Upload one or more PDF or TXT documents
3. Click `Import and Rebuild Knowledge Base`
4. Wait for indexing to finish
5. Ask questions in the chat input

## Configuration

You can tune chunking and retrieval behavior through environment variables:

```powershell
$env:CHUNK_SIZE=800
$env:CHUNK_OVERLAP=150
$env:TOP_K=5
$env:EMBEDDING_MODEL_NAME="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
streamlit run app.py
```

Available variables:

- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `MIN_CHUNK_LENGTH`
- `TOP_K`
- `MAX_ANSWER_SENTENCES`
- `EMBEDDING_MODEL_NAME`

## Interview Talking Points

If you want to present this project in an interview, you can describe it like this:

> I built a local RAG document Q&A assistant that supports PDF and TXT ingestion, document cleaning, chunking, FAISS-based semantic retrieval, and a Streamlit user interface. The main goal was to implement the full retrieval pipeline in a lightweight, CPU-friendly way and keep the system modular enough for future upgrades such as local LLM generation, reranking, and multi-file knowledge base management.

## Future Improvements

- Integrate a local LLM through Ollama for more natural answer generation
- Add reranking to improve retrieval precision
- Support more file types such as Markdown and Word documents
- Add evaluation metrics for retrieval quality
- Highlight cited chunks directly inside the interface

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.
