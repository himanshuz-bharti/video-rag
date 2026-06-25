import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Provider configuration: "gemini" or "ollama"
PROVIDER: str = "ollama" 

# Gemini (Cloud) Configuration
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
VISION_MODEL: str = "gemini-2.5-flash"
EMBED_MODEL: str = "gemini-embedding-2"
EMBED_DIM: int = 768  # Both gemini-embedding-2 and nomic-embed-text are 768-dimensional

# Ollama (Local) Configuration
OLLAMA_VISION_MODEL: str = "gemma3:4b"  # Your pulled vision model (or "llama3.2-vision")
OLLAMA_EMBED_MODEL: str = "nomic-embed-text" # Standard high-quality local embedding model

FRAME_INTERVAL_SECONDS: int = 10
JPEG_QUALITY: int = 85
VIDEO_FORMAT: str = "bestvideo[height<=720][vcodec^=avc1]+bestaudio[ext=m4a]/best[height<=720][vcodec^=avc1]/bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]"

COLLECTION_NAME: str = "video-frames"
TOP_K_RESULTS: int = 4

PROJECT_ROOT: Path = Path(__file__).resolve().parent
FRAMES_DIR: Path = PROJECT_ROOT / "frames"
VIDEO_DIR: Path = PROJECT_ROOT / "videos"
CHROMA_DB_DIR: Path = PROJECT_ROOT / "chroma_db"
SSIM_THRESHOLD: float = 0.50
HASH_THRESHOLD: int = 12
HIST_THRESHOLD_SAVED: float = 0.70
HIST_THRESHOLD_PREV: float = 0.95


