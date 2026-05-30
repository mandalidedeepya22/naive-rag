# Naive RAG Engine with Gemini and Pinecone

A complete, beginner-friendly, fully working Naive Retrieval-Augmented Generation (RAG) web application built using Python, Antigravity, Gemini API (`google-genai`), Pinecone SDK v3+, LangChain, and `python-dotenv`.

## 🏗️ Architecture

```
User Query
↓
Embedding Generation
↓
Pinecone Search
↓
Retrieve Top 3 Chunks
↓
LLM Generation
↓
Final Answer
```

1. **Document Loading**: Text documents are loaded from the `data/` directory.
2. **Text Chunking**: Documents are split into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`.
3. **Embedding Generation**: Text chunks are embedded into dense vectors using Gemini's `models/text-embedding-004` (768 dimensions).
4. **Vector Storage**: Chunk embeddings are stored in Pinecone using a serverless index with cosine metric similarity under the `default` namespace.
5. **Retrieval**: When a query is asked, it is embedded using the same model, and the top 3 most similar chunks are fetched from Pinecone.
6. **Grounded Generation**: The retrieved chunks are formatted into context along with the user query, and an answer is generated using Gemini's `gemini-2.5-flash` model.

## 📋 Prerequisites

- Python 3.8 or higher
- Gemini API Key
- Pinecone API Key

## 🚀 Quick Start Setup

### Step 1: Install Dependencies

Navigate to the project directory and install the requirements:

```bash
cd naive-rag
pip install -r requirements.txt
```

### Step 2: Configure Environment

Copy the example environment template to create your `.env` configuration file:

```bash
cp .env.example .env
```

Open `.env` and fill in your API credentials:

```ini
GEMINI_API_KEY=your_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=naive-rag-index
```

### Step 3: Ingest Documents

Place your text files (e.g. `.txt`) in the `data/` folder (a standard `sample.txt` is already included). Run the ingestion script:

```bash
python ingest.py
```

The script will output progress logs in this exact sequence:
```
Loading documents...
Splitting documents...
Generating embeddings...
Uploading vectors...
Upload complete
```

### Step 4: Run the Application

Start the local web server and launch the clean Antigravity web interface:

```bash
python app.py
```

This will run the server on `http://localhost:8080` and automatically open your default browser.

## 💡 Example Queries

Try the following sample questions once the `sample.txt` document is ingested:

- *What is Retrieval-Augmented Generation?*
- *What does RAG stand for?*
- *Explain what embeddings are.*
- *How do vector databases store and retrieve high-dimensional vectors?*
- *What dimensions are produced by the Gemini text-embedding-004 model?*

## 📁 Project Structure

```
naive-rag/
│
├── app.py                 # Main server serving the Antigravity web interface
├── ingest.py              # Ingestion pipeline processing documents
├── requirements.txt       # Python project dependencies
├── .env.example           # Example environment variable file
├── README.md              # Project documentation
├── data/
│   └── sample.txt         # Pre-configured sample knowledge document
├── utils/
│   ├── __init__.py        # Package initialization
│   ├── embeddings.py      # Gemini embedding generation
│   ├── pinecone_db.py     # Pinecone index creation and search
│   ├── retriever.py       # Context retrieval logic
│   └── generator.py       # Grounded text generation
└── assets/
    └── .gitkeep           # Assets placeholder
```
