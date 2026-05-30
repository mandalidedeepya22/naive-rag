"""Retriever module for fetching relevant document chunks from Pinecone."""

from utils.embeddings import get_embedding
from utils.pinecone_db import search_vectors


def retrieve_chunks(query, index, top_k=3):
    """Retrieve the most relevant document chunks for a given query.
    
    Args:
        query: The user's query.
        index: Pinecone index object.
        top_k: Number of results to retrieve (default: 3).
        
    Returns:
        List of dictionaries containing 'text', 'source', and 'score'.
    """
    query_vector = get_embedding(query)
    matches = search_vectors(index, query_vector, top_k=top_k)
    
    chunks = []
    for match in matches:
        if "metadata" in match and "text" in match["metadata"]:
            chunks.append({
                "text": match["metadata"]["text"],
                "source": match["metadata"].get("source", "unknown"),
                "score": match.get("score", 0.0)
            })
    return chunks