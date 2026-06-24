import sys
import os
from pathlib import Path

# Add parent directory and ingest directory to Python path for seamless imports
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))
ingest_dir = parent_dir / "ingest"
if str(ingest_dir) not in sys.path:
    sys.path.append(str(ingest_dir))

import config
from download import download_video, extract_video_frame
from describe import describe_frame
from embedding import embed_text

# Disable Chroma telemetry warnings
os.environ['CHROMA_TELEMETRY'] = "false"
import chromadb

def ingest(video_url: str, start_frame_ct: int = 0, existing_hists: list = None, pause_event = None, on_frame_extracted_callback = None, on_status_update_callback = None):
    """
    Ingestion Pipeline:
    1. Downloads the YouTube video using yt-dlp.
    2. Extracts keyframes starting from start_frame_ct using OpenCV.
    3. Describes each frame using Gemini Vision model.
    4. Generates a text embedding of the description.
    5. Saves the text, embedding, and metadata (timestamps/paths) into ChromaDB.
    """
    print(f"=== Starting Ingestion Pipeline for Video: {video_url} (Start frame: {start_frame_ct}) ===")
    
    # Step 1: Download video
    print("\n--- Step 1: Downloading Video ---")
    video_path_str = download_video(video_url, config.VIDEO_DIR)
    if not video_path_str:
        print("Error: Video download failed.")
        return [], existing_hists or [], start_frame_ct, False
        
    video_path = Path(video_path_str)
    video_id = video_path.stem
    print(f"Video downloaded successfully to: {video_path}")
    
    # Step 2: Extract keyframes using OpenCV
    print("\n--- Step 2: Extracting Keyframes ---")
    frames, saved_hists, last_frame_ct, completed = extract_video_frame(
        video_path,
        start_frame_ct=start_frame_ct,
        existing_hists=existing_hists,
        pause_event=pause_event,
        on_frame_extracted_callback=on_frame_extracted_callback
    )
    
    if not frames:
        print("No new frames extracted in this run.")
        return [], saved_hists, last_frame_ct, completed
        
    print(f"Extracted {len(frames)} new frames in this batch.")
    
    # Step 3: Describe, Embed, and Index
    print("\n--- Step 3: Generating Descriptions and Indexing in ChromaDB ---")
    
    if on_status_update_callback is not None:
        on_status_update_callback("indexing", f"Keyframe extraction paused/completed. Describing and indexing {len(frames)} keyframes...")
    
    # Initialize ChromaDB persistent client
    chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DB_DIR))
    
    collection = chroma_client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=None
    )
    
    # Clean any old index for the same video_id only on the initial run
    if start_frame_ct == 0:
        try:
            collection.delete(where={"video_id": video_id})
            print(f"Cleared existing index entries for video ID: {video_id}")
        except Exception as e:
            pass
        
    # Process and index each frame
    for idx, frame in enumerate(frames):
        frame_path = frame["frame_path"]
        timestamp = frame["timestamp"]
        frame_ct = frame["frame_ct"]
        
        # Minutes/Seconds formatting for log readability
        m, s = divmod(int(timestamp), 60)
        timestamp_str = f"{m:02d}:{s:02d}"
        
        print(f"\n[{idx+1}/{len(frames)}] Processing frame at {timestamp_str}...")
        if on_status_update_callback is not None:
            on_status_update_callback("indexing", f"Indexing frame {idx+1}/{len(frames)} (at timestamp {timestamp_str})...")
        
        # 3a. Describe visual content using the configured Provider
        description = describe_frame(frame_path)
        if not description:
            print(f"Skipping frame {frame_ct} (failed to generate description).")
            continue
        print(f"Description: {description}")
        
        # 3b. Generate text embeddings using the configured Provider
        embedding = embed_text(description)
        if not embedding:
            print(f"Skipping frame {frame_ct} (failed to generate embedding).")
            continue
            
        # 3c. Insert document + embedding + metadata into ChromaDB
        frame_id = f"{video_id}_frame_{frame_ct}"
        collection.add(
            ids=[frame_id],
            embeddings=[embedding],
            documents=[description],
            metadatas=[{
                "video_id": video_id,
                "timestamp": timestamp,
                "timestamp_str": timestamp_str,
                "frame_ct": frame_ct,
                "frame_path": frame_path
            }]
        )
        print(f"Successfully indexed frame {frame_id} in ChromaDB.")
        
    print("\n=== Ingestion Run Completed ===")
    return frames, saved_hists, last_frame_ct, completed

if __name__ == "__main__":
    # Test pipeline on the
    test_url = "https://youtu.be/r6k3NdKoMX8?si=RqFqa7Jpt1MxyEUV"
    ingest(test_url)
