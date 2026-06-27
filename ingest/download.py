from decimal import DecimalTuple
import os
import time
import sys
from pathlib import Path
from PIL import Image
import imagehash
import ssl

# Bypass python SSL certificate verification globally to avoid [SSL: UNEXPECTED_EOF_WHILE_READING] crashes in restricted cloud networks (Hugging Face)
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

import yt_dlp
from google import genai
import cv2
from google.genai import types

import shutil
import static_ffmpeg

# Only download and configure static_ffmpeg if ffmpeg is not already in the system PATH.
# This prevents redundant download hangs inside containerized environments where ffmpeg is preinstalled.
if not shutil.which("ffmpeg"):
    try:
        static_ffmpeg.add_paths()
    except Exception as e:
        print(f"Warning: static_ffmpeg failed to configure paths: {e}")

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
        'nocheckcertificate': True,
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
        'nocheckcertificate': True,
        'extractor_args': {
            'youtube': {
                'player_client': ['default', '-android_sdkless']
            }
        },
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-us,en;q=0.5',
            'Sec-Fetch-Mode': 'navigate',
        }
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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {fps}, Total Frames: {total_frames}")

    # Determine target number of keyframes
    target_keyframes = 10
    
    # Divide video into 10 equal intervals
    interval_size = max(1, total_frames // target_keyframes)
    extracted = []
    saved_hists = list(existing_hists) if existing_hists is not None else []

    # If resuming, we only extract keyframes in intervals that start after start_frame_ct
    for i in range(target_keyframes):
        # Calculate the frame range for this interval
        start_range = i * interval_size
        end_range = min(start_range + interval_size, total_frames)

        # If resuming from checkpoint, skip intervals we already processed
        if start_range < start_frame_ct:
            continue

        # Check if pause event is set
        if pause_event is not None and pause_event.is_set():
            print(f"Extraction paused by user event at interval {i}")
            break

        # Check if we already have 10 keyframes (limit)
        if len(saved_hists) >= 10:
            break

        # Scan this interval starting from the middle to find a bright, high-quality frame
        # (This avoids transition boundaries at the edges of the interval)
        mid_frame = start_range + (interval_size // 2)
        
        # We try to find a valid frame starting from mid_frame, scanning up to end_range
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        current_frame_idx = mid_frame
        
        saved_frame = False
        skipped_due_to_similarity = False
        while current_frame_idx < end_range:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Check brightness of frame to avoid black screens
            h, w = frame.shape[:2]
            small_w = 256
            small_h = int(h * (small_w / w))
            frame_small = cv2.resize(frame, (small_w, small_h))
            
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            mean_brightness = gray.mean()
            
            if mean_brightness >= 15.0:
                # Compute Perceptual Hash (pHash) to evaluate structural similarity
                pil_img = Image.fromarray(cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB))
                current_hash = imagehash.phash(pil_img)
                
                # Check similarity against all previously saved keyframes
                should_save = False
                min_distance = 64
                
                if not saved_hists:
                    should_save = True
                else:
                    min_distance = min(current_hash - saved_hash for saved_hash in saved_hists)
                    if min_distance > config.HASH_THRESHOLD:
                        should_save = True
                    else:
                        skipped_due_to_similarity = True
                
                if should_save:
                    # Found a valid bright and visually unique frame! Save it.
                    timestamp = current_frame_idx / fps
                    frame_filename = f"{current_frame_idx:06d}.jpg"
                    frame_path = config.FRAMES_DIR / frame_filename

                    save_frame = frame
                    if w > 768:
                        scale = 768 / w
                        save_frame = cv2.resize(frame, (768, int(h * scale)))

                    cv2.imwrite(str(frame_path), save_frame, [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY])
                    saved_hists.append(current_hash)

                    print(f"Interval {i+1}/{target_keyframes}: Saved keyframe {current_frame_idx} at {timestamp:.2f}s (brightness: {mean_brightness:.2f}, min pHash distance: {min_distance})")
                    
                    frame_info = {
                        "frame_path": str(frame_path),
                        "timestamp": timestamp,
                        "frame_ct": current_frame_idx
                    }
                    extracted.append(frame_info)
                    
                    if on_frame_extracted_callback is not None:
                        on_frame_extracted_callback(frame_info)
                        
                    saved_frame = True
                    break  # Move to the next interval
                
            current_frame_idx += 1

        # Fallback: if all frames in the second half of the interval were black/dark (and NOT skipped due to similarity),
        # just save the middle frame anyway so we don't miss the interval completely
        if not saved_frame and not skipped_due_to_similarity:
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret, frame = cap.read()
            if ret:
                timestamp = mid_frame / fps
                frame_filename = f"{mid_frame:06d}.jpg"
                frame_path = config.FRAMES_DIR / frame_filename

                h, w = frame.shape[:2]
                save_frame = frame
                if w > 768:
                    scale = 768 / w
                    save_frame = cv2.resize(frame, (768, int(h * scale)))

                cv2.imwrite(str(frame_path), save_frame, [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY])
                
                # Save computed hash as fallback
                small_w = 256
                small_h = int(h * (small_w / w))
                frame_small = cv2.resize(frame, (small_w, small_h))
                pil_img = Image.fromarray(cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB))
                current_hash = imagehash.phash(pil_img)
                saved_hists.append(current_hash)

                print(f"Interval {i+1}/{target_keyframes} (Fallback): Saved keyframe {mid_frame} at {timestamp:.2f}s")
                
                frame_info = {
                    "frame_path": str(frame_path),
                    "timestamp": timestamp,
                    "frame_ct": mid_frame
                }
                extracted.append(frame_info)
                
                if on_frame_extracted_callback is not None:
                    on_frame_extracted_callback(frame_info)

    cap.release()
    print(f"Extracted {len(extracted)} keyframes in this run. Total: {len(saved_hists)}") 
    return extracted, saved_hists, total_frames, True
            
if __name__ == "__main__":
    downloaded_file = download_video('https://youtu.be/frXG1mo_OBQ?si=I7P6e5fRTj7a1zHD', config.VIDEO_DIR)
    if downloaded_file:
        extracted = extract_video_frame(Path(downloaded_file))
        print(extracted[:5])
    else:
        print("Download failed. Cannot extract video frames.")


            

