"""Gemini embeddings module for generating vector representations of text."""

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()


def init_embeddings():
    """Initialize the embeddings model name.
    
    Returns:
        The name of the embedding model.
    """
    return "text-embedding-004"


def get_embedding(text):
    """Generate an embedding vector for the given text using Gemini API.
    
    Args:
        text: The text string to embed.
        
    Returns:
        A list of floats representing the 768-dimensional embedding vector.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY. Please check your .env file.")
    
    client = genai.Client(api_key=api_key)
    
    try:
        # Try text-embedding-004 first as per the prompt instructions
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=text
        )
        return response.embeddings[0].values
    except Exception as e:
        # Fall back to gemini-embedding-2 with 768 dimensions if text-embedding-004 is unavailable
        if "404" in str(e) or "not found" in str(e).lower() or "not_found" in str(e).lower():
            from google.genai import types
            response = client.models.embed_content(
                model="gemini-embedding-2",
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=768)
            )
            return response.embeddings[0].values
        raise e