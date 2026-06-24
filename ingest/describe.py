import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pathlib import Path
from skimage.metrics import structural_similarity
import ollama
import sys

# Add the parent directory to the Python path to import root-level config module
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))
    
import config
load_dotenv()


def describe_frame_gemini(image_path: str, max_retries: int = 5, backoff_factor: float = 2.0) -> str:
    """Describe an image in detail in 5-7 lines using Gemini, with exponential backoff for 429 errors."""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    prompt = "Describe this video frame in less than 20 words or at most 2 sentences. Focus briefly on the main visual content, text, or action."

    delay = 10.0  # start with 10s delay if rate limited
    for attempt in range(max_retries):
        try:
            with open(image_path, "rb") as img:
                image_bytes = img.read()

            response = client.models.generate_content(
                model=config.VISION_MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    prompt 
                ]
            )
            return response.text.strip()
            
        except Exception as e:
            err_msg = str(e)
            is_rate_limit = "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg
            
            if is_rate_limit and attempt < max_retries - 1:
                print(f"[Rate Limit] Exceeded quota on {image_path}. Retrying in {delay} seconds (Attempt {attempt+1}/{max_retries})...")
                time.sleep(delay)
                delay *= backoff_factor
            else:
                print(f"Error describing frame {image_path} with Gemini: {e}")
                return ""
    return ""


def describe_frame_ollama(image_path: str, model_name: str = None) -> str:
    """Describe an image in detail using local Ollama vision model."""
    if model_name is None:
        model_name = config.OLLAMA_VISION_MODEL
        
    try:
        
        prompt = "Describe this video frame in less than 20 words or at most 2 sentences. Focus briefly on the main visual content, text, or action."

        response = ollama.chat(
            model=model_name,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_path]
            }]
        )
        return response['message']['content'].strip()
    except Exception as e:
        print(f"Error describing frame {image_path} using local Ollama: {e}")
        return ""


def describe_frame(image_path: str) -> str:
    """Unified entry point to describe an image based on the configured PROVIDER."""
    if config.PROVIDER == "ollama":
        return describe_frame_ollama(image_path)
    else:
        return describe_frame_gemini(image_path)
        

if __name__ == "__main__":
   
    test_images = ['frames/000600.jpg']
    if test_images:
        print(f"Testing description on {test_images[0]} using provider: {config.PROVIDER}...")
        desc = describe_frame(test_images[0])
        print("Description:\n", desc)
    else:
        print("No frames found to test description.")

    