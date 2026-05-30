"""Pinecone database module for vector storage and similarity search."""

import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()


def init_pinecone():
    """Initialize and return a Pinecone client.
    
    Returns:
        Pinecone client object.
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("Missing PINECONE_API_KEY. Please check your .env file.")
    return Pinecone(api_key=api_key)


def create_index_if_missing(pc, index_name):
    """Create a Pinecone index if it doesn't already exist and wait until it's ready.
    
    Args:
        pc: Pinecone client object.
        index_name: Name of the index to create.
    """
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME is not set in environment variables.")
        
    try:
        existing_indexes = pc.list_indexes().names()
    except AttributeError:
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        # Wait until index becomes ready
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)


def upsert_vectors(index, vectors):
    """Upsert vectors into the Pinecone index under the 'default' namespace.
    
    Args:
        index: Pinecone index object.
        vectors: List of tuples/dicts (id, vector, metadata).
    """
    index.upsert(vectors=vectors, namespace="default")


def search_vectors(index, query_vector, top_k=3):
    """Search for the most similar vectors in the index under the 'default' namespace.
    
    Args:
        index: Pinecone index object.
        query_vector: The embedding vector of the query.
        top_k: Number of results to return.
        
    Returns:
        List of matches containing id, score, and metadata.
    """
    response = index.query(
        namespace="default",
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    return response["matches"]