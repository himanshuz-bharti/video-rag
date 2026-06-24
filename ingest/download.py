from decimal import DecimalTuple
import os
import time
import sys
from pathlib import Path
from PIL import Image
import imagehash

import yt_dlp
from google import genai
import cv2
from google.genai import types

import static_ffmpeg
static_ffmpeg.add_paths()

# Add the parent directory to the Python path to import root-level config module
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

import config
os.environ['CHROMA_TELEMETRY']="false"
import chromadb

def get_video_info(url:str)->dict:
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False, 
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url,download=False)

            metadata = {
                "id": info.get("id"),
                "title": info.get("title"),
                "uploader": info.get("uploader"),
                "upload_date": info.get("upload_date"),
                "duration": info.get("duration"),
                "description": info.get("description"),
                "url": info.get("webpage_url"),
                "view_count": info.get("view_count"),
                "like_count": info.get("like_count"),
                "comment_count": info.get("comment_count"),
                "category": info.get("category"),
                "tags": info.get("tags"),
                "channel_url":info.get(""),
                "channel_id":info.get("channel_id"  ),
            }
            return metadata
        except Exception as e:
            print(f"Error getting video info: {e}")
            return {}

def download_video(url:str,output_dir: str)->str:
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True,parents=True)

    format_selector = config.VIDEO_FORMAT
    
    
    ydl_opts = {
        'format':format_selector,
        'outtmpl':str(out_path / '%(id)s.%(ext)s'),
        'quiet':False,
        'no_warnings':True,
        'noplaylist':True,
        # 'writesubtitles':True,
        # 'writeautomaticsub':True,
        # 'subtitleslangs':['en'],
        # 'subtitlesformat':'vtt',
        }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url,download=True)
            filename = ydl.prepare_filename(info)
            return filename
    except Exception as e:
        print(f"Failed video download (checking for Windows lock recovery): {e}")
        # Try to recover from Windows WinError 32 lock error during merge rename
        try:
            for temp_file in out_path.glob("*.temp.mp4"):
                target_file = temp_file.with_name(temp_file.name.replace(".temp.mp4", ".mp4"))
                print(f"Found merged temp file: {temp_file}. Attempting to recover...")
                
                # Retry renaming up to 5 times with 1-second delays for lock to release
                for attempt in range(5):
                    try:
                        time.sleep(1.0)
                        if temp_file.exists():
                            # If target already exists, delete it first to prevent rename failure
                            if target_file.exists():
                                target_file.unlink()
                            temp_file.rename(target_file)
                            print(f"Successfully recovered and renamed locked file: {target_file}")
                            
                            # Clean up separate audio/video chunks left behind
                            base_name = temp_file.stem.split('.')[0]
                            for part_file in out_path.glob(f"{base_name}.f*"):
                                try:
                                    part_file.unlink()
                                except Exception:
                                    pass
                            return str(target_file)
                    except Exception as rename_err:
                        print(f"Rename attempt {attempt+1} failed: {rename_err}")
        except Exception as recovery_err:
            print(f"Recovery failed: {recovery_err}")
            
        return ""
    
# info = get_video_info('https://youtu.be/uZGDO0L-Dr4?si=YXVogHhC8GR8dqNV')
# print(info)

def extract_video_frame(video_path: Path, start_frame_ct: int = 0, existing_hists: list = None, pause_event = None, on_frame_extracted_callback = None):
    config.FRAMES_DIR.mkdir(exist_ok=True, parents=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")
    dur = total_frames / fps if fps > 0 else 0
    print(f"Duration: {dur}")
    
    # We sample a candidate frame every 2 seconds and verify structural changes
    eval_interval = max(1, int(2 * fps)) 
    print(f"Evaluating Image hash every {eval_interval} frames (~2.0 seconds)")

    extracted = []
    frame_ct = 0
    saved_hists = list(existing_hists) if existing_hists is not None else []
    prev_hist = None
    reached_end = True

    # If resuming from a checkpoint, set the frame position
    if start_frame_ct > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_ct)
        frame_ct = start_frame_ct
        print(f"Resuming frame extraction from frame {start_frame_ct}")

    while True:
        # Check if pause event is set by the web server
        if pause_event is not None and pause_event.is_set():
            print(f"Extraction paused by user event at frame {frame_ct}")
            reached_end = False
            break

        ret, frame = cap.read()
        if not ret:
            break

        if frame_ct % eval_interval == 0:
            if len(saved_hists) >= 10:
                print("Reached maximum limit of 10 keyframes. Stopping extraction.")
                break

            h, w = frame.shape[:2]
            
            # Downscale frame for fast computation and robustness
            small_w = 256
            small_h = int(h * (small_w / w))
            frame_small = cv2.resize(frame, (small_w, small_h))
            
            # Convert to HSV color space
            hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)
            
            # Calculate 2D Hue-Saturation histogram (16 bins each)
            current_hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
            cv2.normalize(current_hist, current_hist)

            should_save = False
            max_sim_saved = 0.0
            sim_prev = 1.0

            if not saved_hists:
                # Always save the first frame
                should_save = True
            else:
                # Compare against all previously saved histograms to enforce global uniqueness
                max_sim_saved = max(cv2.compareHist(current_hist, saved_hist, cv2.HISTCMP_CORREL) for saved_hist in saved_hists)
                sim_prev = cv2.compareHist(current_hist, prev_hist, cv2.HISTCMP_CORREL)
                if max_sim_saved < config.HIST_THRESHOLD_SAVED and sim_prev < config.HIST_THRESHOLD_PREV:
                    should_save = True

            if should_save:
                timestamp = frame_ct / fps
                frame_filename = f"{frame_ct:06d}.jpg"
                frame_path = config.FRAMES_DIR / frame_filename

                # Scale frame to a max width of 768px for storage
                save_frame = frame
                if w > 768:
                    scale = 768 / w
                    save_frame = cv2.resize(frame, (768, int(h * scale)))

                cv2.imwrite(str(frame_path), save_frame, [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY])
                
                print(f"Saved keyframe {frame_ct} at {timestamp:.2f}s (Max Sim Saved: {max_sim_saved:.3f}, Sim Prev: {sim_prev:.3f}) - Dimensions: {save_frame.shape[1]}x{save_frame.shape[0]}")
                
                frame_info = {
                    "frame_path": str(frame_path),
                    "timestamp": timestamp,
                    "frame_ct": frame_ct
                }
                extracted.append(frame_info)
                
                if on_frame_extracted_callback is not None:
                    on_frame_extracted_callback(frame_info)
                
                saved_hists.append(current_hist)

            prev_hist = current_hist
                
                
        frame_ct += 5

    cap.release()
    print(f"Extracted {len(extracted)} keyframes in this run. Total in memory: {len(saved_hists)}") 
    return extracted, saved_hists, frame_ct, reached_end
            
if __name__ == "__main__":
    downloaded_file = download_video('https://youtu.be/frXG1mo_OBQ?si=I7P6e5fRTj7a1zHD', config.VIDEO_DIR)
    if downloaded_file:
        extracted = extract_video_frame(Path(downloaded_file))
        print(extracted[:5])
    else:
        print("Download failed. Cannot extract video frames.")


            

