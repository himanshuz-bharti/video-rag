import os
# If OLLAMA_HOST is set to 0.0.0.0, override it to 127.0.0.1 for the local Python client
# since Windows prevents clients from connecting directly to the 0.0.0.0 bind address.
if os.getenv("OLLAMA_HOST") == "0.0.0.0":
    os.environ["OLLAMA_HOST"] = "127.0.0.1"

import sys
import shutil
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks, Header
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Add parent directory and ingest directory to Python path for seamless imports
parent_dir = Path(__file__).resolve().parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))
ingest_dir = parent_dir / "ingest"
if str(ingest_dir) not in sys.path:
    sys.path.append(str(ingest_dir))

import config

load_dotenv()

app = FastAPI(title="Video RAG Dashboard")

# Ensure required directories exist
config.FRAMES_DIR.mkdir(exist_ok=True, parents=True)
config.VIDEO_DIR.mkdir(exist_ok=True, parents=True)

# Mount the frames directory to serve extracted keyframes statically
app.mount("/frames", StaticFiles(directory=str(config.FRAMES_DIR)), name="frames")

import threading

# Thread-safe event to signal pause/stop to the extraction thread
PAUSE_EVENT = threading.Event()

# Global session state tracker
SESSION_STATE = {
    "status": "idle",           # "idle", "processing", "paused", "success", "error"
    "message": "",
    "youtube_url": "",
    "start_frame_ct": 0,
    "saved_hists": [],
    "extracted_frames": [],      # List of currently extracted frame objects
    "api_key": None,
    "provider": None
}

class ProcessRequest(BaseModel):
    youtube_url: str

class QueryRequest(BaseModel):
    query_text: str

def run_ingestion_background(youtube_url: str):
    global SESSION_STATE
    try:
        SESSION_STATE["status"] = "processing"
        SESSION_STATE["message"] = "Downloading video from YouTube..."
        SESSION_STATE["youtube_url"] = youtube_url
        
        # Import dynamically to ensure path setup takes effect
        from ingest import ingest
        
        SESSION_STATE["message"] = "Processing video, extracting keyframes, and generating descriptions..."
        
        start_frame = SESSION_STATE["start_frame_ct"]
        hists = SESSION_STATE["saved_hists"]
        api_key = SESSION_STATE.get("api_key")
        provider = SESSION_STATE.get("provider")
        
        # Define a callback to populate extracted_frames list in real time
        def on_frame_extracted(frame_info):
            p = Path(frame_info["frame_path"])
            frame_info["image_url"] = f"/frames/{p.name}"
            # Check if this frame is already in list to prevent duplicate appends
            if not any(f["frame_ct"] == frame_info["frame_ct"] for f in SESSION_STATE["extracted_frames"]):
                SESSION_STATE["extracted_frames"].append(frame_info)
        
        # Define a callback to update status and message dynamically
        def on_status_update(status, message):
            SESSION_STATE["status"] = status
            SESSION_STATE["message"] = message
        
        # Run the ingestion run
        new_frames, saved_hists, last_frame_ct, completed = ingest(
            youtube_url,
            start_frame_ct=start_frame,
            existing_hists=hists,
            pause_event=PAUSE_EVENT,
            on_frame_extracted_callback=on_frame_extracted,
            on_status_update_callback=on_status_update,
            provider=provider,
            api_key=api_key
        )
        
        # Update checkpoint progress state
        SESSION_STATE["saved_hists"] = saved_hists
        SESSION_STATE["start_frame_ct"] = last_frame_ct
        
        if completed:
            SESSION_STATE["status"] = "success"
            SESSION_STATE["message"] = f"Ingestion completed! Total {len(SESSION_STATE['extracted_frames'])} distinct keyframes indexed."
        else:
            if PAUSE_EVENT.is_set():
                SESSION_STATE["status"] = "paused"
                SESSION_STATE["message"] = f"Ingestion paused. {len(SESSION_STATE['extracted_frames'])} keyframes indexed so far."
            else:
                # Reached cap/max of 10 keyframes or finished otherwise
                SESSION_STATE["status"] = "success"
                SESSION_STATE["message"] = f"Ingestion completed (saved max limit of {len(SESSION_STATE['extracted_frames'])} keyframes)."
    except Exception as e:
        SESSION_STATE["status"] = "error"
        SESSION_STATE["message"] = f"Failed to index video: {str(e)}"
        print(f"Background Ingestion Error: {e}")

@app.get("/")
def serve_index():
    """Serve the single-page RAG frontend dashboard."""
    index_path = parent_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(str(index_path))

def sanitize_value(v):
    """Recursively convert NumPy types and other non-serializable objects to python primitives."""
    if hasattr(v, "tolist") and callable(getattr(v, "tolist")):
        try:
            return v.tolist()
        except Exception:
            pass
    if hasattr(v, "item") and callable(getattr(v, "item")):
        try:
            return v.item()
        except Exception:
            pass
    if isinstance(v, dict):
        return {str(k): sanitize_value(vv) for k, vv in v.items()}
    if isinstance(v, list):
        return [sanitize_value(vv) for vv in v]
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)

@app.get("/api/status")
def get_status():
    """Retrieve the current processing status and progress."""
    try:
        # Strip non-serializable fields (NumPy array list saved_hists) to prevent jsonable_encoder crash
        status_copy = {}
        for k, v in SESSION_STATE.items():
            if k == "saved_hists":
                continue
            status_copy[k] = sanitize_value(v)
        return status_copy
    except Exception as e:
        print(f"Error serializing status in get_status: {e}")
        # Return a safe fallback status dictionary so polling never breaks with 500 error
        return {
            "status": SESSION_STATE.get("status", "error"),
            "message": f"Serialization Error: {str(e)}",
            "youtube_url": SESSION_STATE.get("youtube_url", ""),
            "start_frame_ct": SESSION_STATE.get("start_frame_ct", 0),
            "extracted_frames": []
        }

@app.post("/api/process")
def process_video(
    req: ProcessRequest,
    background_tasks: BackgroundTasks,
    x_api_key: str = Header(default=None),
    x_provider: str = Header(default=None)
):
    """Start downloading and indexing a video in a background thread."""
    global SESSION_STATE
    if SESSION_STATE["status"] == "processing":
        raise HTTPException(status_code=400, detail="Another video is already being processed.")
    
    # If starting a fresh video, reset session state and clean folders/DB
    if req.youtube_url != SESSION_STATE["youtube_url"]:
        SESSION_STATE["start_frame_ct"] = 0
        SESSION_STATE["saved_hists"] = []
        SESSION_STATE["extracted_frames"] = []
        
        # Clear previous video files, frames, and database index
        try:
            import chromadb
            if config.VIDEO_DIR.exists():
                shutil.rmtree(config.VIDEO_DIR)
            config.VIDEO_DIR.mkdir(parents=True, exist_ok=True)
            
            if config.FRAMES_DIR.exists():
                shutil.rmtree(config.FRAMES_DIR)
            config.FRAMES_DIR.mkdir(parents=True, exist_ok=True)
            
            chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DB_DIR))
            try:
                chroma_client.delete_collection(name=config.COLLECTION_NAME)
            except Exception:
                pass
            chroma_client.get_or_create_collection(name=config.COLLECTION_NAME)
            print("Cleared previous video files, frames, and index for new ingestion.")
        except Exception as clean_err:
            print(f"Error cleaning previous session data: {clean_err}")
        
    SESSION_STATE["api_key"] = x_api_key
    SESSION_STATE["provider"] = x_provider
    
    PAUSE_EVENT.clear()
    background_tasks.add_task(run_ingestion_background, req.youtube_url)
    return {"message": "Ingestion task launched."}

@app.post("/api/pause")
def pause_extraction():
    """Signal the ingestion background thread to pause frame extraction."""
    global SESSION_STATE
    if SESSION_STATE["status"] != "processing":
        raise HTTPException(status_code=400, detail="Pipeline is not currently processing.")
    
    PAUSE_EVENT.set()
    SESSION_STATE["message"] = "Pausing ingestion, wrapping up current frame..."
    return {"message": "Pause request sent."}

@app.post("/api/resume")
def resume_extraction(
    background_tasks: BackgroundTasks,
    x_api_key: str = Header(default=None),
    x_provider: str = Header(default=None)
):
    """Resume indexing the current video from where it left off."""
    global SESSION_STATE
    if SESSION_STATE["status"] != "paused" and SESSION_STATE["status"] != "error":
        raise HTTPException(status_code=400, detail="Pipeline is not in a paused or errored state.")
        
    if not SESSION_STATE["youtube_url"]:
        raise HTTPException(status_code=400, detail="No video is currently loaded in session.")
        
    if x_api_key:
        SESSION_STATE["api_key"] = x_api_key
    if x_provider:
        SESSION_STATE["provider"] = x_provider
        
    PAUSE_EVENT.clear()
    background_tasks.add_task(run_ingestion_background, SESSION_STATE["youtube_url"])
    return {"message": "Resume task launched."}

@app.post("/api/query")
def query_rag(
    req: QueryRequest,
    x_api_key: str = Header(default=None),
    x_provider: str = Header(default=None)
):
    """Query the vector database and generate an answer with Gemini."""
    if not req.query_text.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")
    
    api_key = x_api_key or SESSION_STATE.get("api_key") or os.getenv("GEMINI_API_KEY")
    provider = x_provider or SESSION_STATE.get("provider") or config.PROVIDER
    
    try:
        from query.query import VideoRAGQuery
        rag = VideoRAGQuery(provider=provider, api_key=api_key)
        answer, sources = rag.answer_query(req.query_text)
        
        # Map absolute frame paths to relative URLs served by FastAPI
        for source in sources:
            p = Path(source["frame_path"])
            source["image_url"] = f"/frames/{p.name}"
            
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/cleanup")
def cleanup_session():
    """Delete all local files and clear the vector database."""
    global SESSION_STATE
    PAUSE_EVENT.clear()
    try:
        # 1. Clear directories
        if config.FRAMES_DIR.exists():
            for f in config.FRAMES_DIR.iterdir():
                if f.is_file():
                    try:
                        f.unlink()
                    except Exception:
                        pass
                        
        if config.VIDEO_DIR.exists():
            for f in config.VIDEO_DIR.iterdir():
                if f.is_file():
                    try:
                        f.unlink()
                    except Exception:
                        pass
                        
        # 2. Clear ChromaDB Collection
        import chromadb
        try:
            chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DB_DIR))
            try:
                chroma_client.delete_collection(name=config.COLLECTION_NAME)
            except Exception:
                pass
                
            # Recreate empty collection so RAG remains functional
            chroma_client.get_or_create_collection(
                name=config.COLLECTION_NAME,
                embedding_function=None
            )
        except Exception as db_err:
            print(f"ChromaDB client error, database might be corrupted. Recreating: {db_err}")
            # Database is likely locked or corrupted; delete the database directory contents directly
            if config.CHROMA_DB_DIR.exists():
                for f in config.CHROMA_DB_DIR.iterdir():
                    if f.is_file():
                        try:
                            f.unlink()
                        except Exception:
                            pass
            # Try initializing client and recreating collection again on clean slate
            chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DB_DIR))
            chroma_client.get_or_create_collection(
                name=config.COLLECTION_NAME,
                embedding_function=None
            )
        
        # Reset Status
        SESSION_STATE = {
            "status": "idle",
            "message": "Session and databases successfully cleared.",
            "youtube_url": "",
            "start_frame_ct": 0,
            "saved_hists": [],
            "extracted_frames": [],
            "api_key": None,
            "provider": None
        }
        return {"message": "Successfully deleted all video frames, videos, and cleared ChromaDB."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Disable reload in production/container environments to prevent file system writes from restarting the server.
    reload = os.getenv("UVICORN_RELOAD", "false").lower() == "true"
    uvicorn.run("main_web:app", host="0.0.0.0", port=8000, reload=reload)
