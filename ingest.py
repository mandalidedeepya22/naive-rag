"""Document ingestion script for Naive RAG."""

import os
import sys
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.embeddings import get_embedding
from utils.pinecone_db import init_pinecone, create_index_if_missing, upsert_vectors

load_dotenv()

# Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
DATA_DIR = "data"


def load_documents():
    """Load all .txt files from the data directory.
    
    Returns:
        List of dicts containing text content and filename.
    """
    documents = []
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        return documents
        
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(DATA_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():
                    documents.append({
                        "text": content,
                        "filename": filename
                    })
    return documents


def split_documents(documents):
    """Split documents into chunks.
    
    Args:
        documents: List of document dicts.
        
    Returns:
        List of chunk dicts.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    chunks = []
    for doc in documents:
        split_texts = text_splitter.split_text(doc["text"])
        for i, chunk_text in enumerate(split_texts):
            chunks.append({
                "text": chunk_text,
                "filename": doc["filename"],
                "chunk_index": i
            })
    return chunks


def generate_embeddings(chunks):
    """Generate embeddings for each chunk.
    
    Args:
        chunks: List of chunk dicts.
        
    Returns:
        List of vector tuples for Pinecone.
    """
    vectors = []
    for chunk in chunks:
        embedding = get_embedding(chunk["text"])
        vector_id = f"{chunk['filename']}_{chunk['chunk_index']}"
        metadata = {
            "text": chunk["text"],
            "source": chunk["filename"]
        }
        vectors.append((vector_id, embedding, metadata))
    return vectors


def main():
    """Main ingestion pipeline."""
    try:
        # Check for required API keys
        if not os.getenv("GEMINI_API_KEY"):
            print("Missing API key. Please check your .env file.")
            sys.exit(1)
            
        if not os.getenv("PINECONE_API_KEY"):
            print("Missing API key. Please check your .env file.")
            sys.exit(1)
            
        index_name = os.getenv("PINECONE_INDEX_NAME")
        if not index_name:
            print("Pinecone index name not specified in environment.")
            sys.exit(1)
            
        print("Loading documents...")
        documents = load_documents()
        if not documents:
            print("No documents found in data directory.")
            return
            
        print("Splitting documents...")
        chunks = split_documents(documents)
        
        print("Generating embeddings...")
        vectors = generate_embeddings(chunks)
        
        # Initialize Pinecone and create index if missing
        pc = init_pinecone()
        create_index_if_missing(pc, index_name)
        index = pc.Index(index_name)
        
        print("Uploading vectors...")
        # Batch upload vectors in size of 100 for safety
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            upsert_vectors(index, batch)
            
        print("Upload complete")
        
    except Exception as e:
        print(f"Error during ingestion: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()