import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from google import genai
import ollama

# Add the parent directory to the Python path to import root-level config module
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

import config

load_dotenv()

def embed_text_gemini(text: str, api_key: str = None) -> list[float]:
    """
    Generate text embeddings using Google's text-embedding-004 model (Cloud API).
    This model produces high-quality 768-dimensional embeddings.
    """
    active_key = api_key or os.getenv("GEMINI_API_KEY")
    if not active_key:
        print("[Error] Gemini API key is missing for embed_text.")
        return []
    client = genai.Client(api_key=active_key)
    model_name = config.EMBED_MODEL
    try:
        response = client.models.embed_content(
            model=model_name,
            contents=text
        )
        # Extract and return the float list of embeddings
        return response.embeddings[0].values
    except Exception as e:
        print(f"Error generating Gemini embedding: {e}")
        return []

def embed_text_ollama(text: str, model_name: str = None) -> list[float]:
    """
    Generate text embeddings using local Ollama model (e.g. nomic-embed-text).
    Requires Ollama running and the model pulled: `ollama pull nomic-embed-text`.
    """
    if model_name is None:
        model_name = config.OLLAMA_EMBED_MODEL
        
    try:
        response = ollama.embeddings(
            model=model_name,
            prompt=text
        )
        return response['embedding']
    except Exception as e:
        print(f"Error generating Ollama embedding: {e}")
        return []


def embed_text(text: str, provider: str = None, api_key: str = None) -> list[float]:
    """Unified entry point to embed text based on the configured PROVIDER."""
    active_provider = provider or config.PROVIDER
    if active_provider == "ollama":
        return embed_text_ollama(text)
    else:
        return embed_text_gemini(text, api_key=api_key)


if __name__ == "__main__":
    test_text = "The video frame shows a dark background with Python code."
    
    print(f"Testing Embeddings with provider: {config.PROVIDER}...")
    embedding = embed_text(test_text)
    if embedding:
        print(f"Embedding generated successfully!")
        print(f"Dimensions: {len(embedding)}")
        print(f"Preview (first 5 values): {embedding[:5]}\n")