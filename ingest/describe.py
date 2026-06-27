import os
# If OLLAMA_HOST is set to 0.0.0.0, override it to 127.0.0.1 for the local Python client
if os.getenv("OLLAMA_HOST") == "0.0.0.0":
    os.environ["OLLAMA_HOST"] = "127.0.0.1"

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


def describe_frame_gemini(image_path: str, api_key: str = None, max_retries: int = 5, backoff_factor: float = 2.0) -> str:
    """Describe the given image in **4–5 detailed sentences**. Your description should be factual and based only on what is visible in the image.

Include the following details:

* The overall scene or environment (indoors/outdoors, location, background, weather, lighting, etc.).
* The number of people visible in the image.
* For each person, describe their approximate age group, gender (if visually apparent), clothing, accessories, hairstyle, facial expression, posture, and any notable actions.
* Mention any important objects, vehicles, animals, or landmarks present.
* Describe the spatial arrangement of the people and objects (e.g., standing, sitting, left/right, foreground/background).
* If any text, signs, logos, or screens are visible, mention them.

Do not make assumptions or infer information that is not directly visible. If a detail is unclear, state that it is not clearly visible rather than guessing.
 """
    active_key = api_key or os.getenv("GEMINI_API_KEY")
    if not active_key:
        print("[Error] Gemini API key is missing for describe_frame.")
        return ""
    client = genai.Client(api_key=active_key)
    
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
    """Describe the given image in **4–5 detailed sentences**. Your description should be factual and based only on what is visible in the image.

Include the following details:

* The overall scene or environment (indoors/outdoors, location, background, weather, lighting, etc.).
* The number of people visible in the image.
* For each person, describe their approximate age group, gender (if visually apparent), clothing, accessories, hairstyle, facial expression, posture, and any notable actions.
* Mention any important objects, vehicles, animals, or landmarks present.
* Describe the spatial arrangement of the people and objects (e.g., standing, sitting, left/right, foreground/background).
* If any text, signs, logos, or screens are visible, mention them.

Do not make assumptions or infer information that is not directly visible. If a detail is unclear, state that it is not clearly visible rather than guessing.
."""
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


def describe_frame(image_path: str, provider: str = None, api_key: str = None) -> str:
    """Unified entry point to describe an image based on the configured PROVIDER."""
    active_provider = provider or config.PROVIDER
    if active_provider == "ollama":
        return describe_frame_ollama(image_path)
    else:
        return describe_frame_gemini(image_path, api_key=api_key)
        

if __name__ == "__main__":
   
    test_images = ['frames/000600.jpg']
    if test_images:
        print(f"Testing description on {test_images[0]} using provider: {config.PROVIDER}...")
        desc = describe_frame(test_images[0])
        print("Description:\n", desc)
    else:
        print("No frames found to test description.")

    