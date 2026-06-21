import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY:str = os.getenv("GEMINI_API_KEY")
VISION_MODEL:str = "gemini-2.5-flash"
EMBED_MODEL:str = ""
EMBED_DIM:int = 3072

FRAME_INTERVAL_SECONDS: int = 30
JPEG_QUALITY: int = 85
VIDEO_FORMAT:str = "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]"

COLLECTION_NAME:str = "video-frames"
TOP_K_RESULTS:int = 4

FRAMES_DIR: Path = Path("frames")
VIDEO_DIR : Path = Path("videos")
