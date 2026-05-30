"""Generator module for producing grounded answers using Gemini API."""

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()


def build_prompt(context, question):
    """Build the exact RAG prompt template from context and question.
    
    Args:
        context: Retrieved context text.
        question: User's question.
        
    Returns:
        Formatted prompt.
    """
    return f"""You are a helpful AI assistant.

Answer only using the provided context.

If the answer is not available in the context say:

"I could not find the answer in the provided documents."

Context:
{context}

Question:
{question}

Answer:"""


def generate_answer(prompt):
    """Generate an answer using Gemini 2.5 Flash.
    
    Args:
        prompt: The formatted prompt string.
        
    Returns:
        The generated answer text.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY. Please check your .env file.")
        
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text