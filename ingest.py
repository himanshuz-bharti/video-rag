from decimal import DecimalTuple
import os
import time
from pathlib import Path

import yt_dlp
from google import genai
import cv2
from google.genai import types

import static_ffmpeg
static_ffmpeg.add_paths()

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
        'writesubtitles':True,
        'writeautomaticsub':True,
        'subtitleslangs':['en'],
        'subtitlesformat':'vtt'
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url,download=True)
            filename = ydl.prepare_filename(info)
            return filename
        except Exception as e:
            print(f"Error downloading video: {e}")
            return ""
    
# info = get_video_info('https://youtu.be/uZGDO0L-Dr4?si=YXVogHhC8GR8dqNV')
# print(info)

def extract_video_frame(video_path:Path):
    config.FRAMES_DIR.mkdir(exist_ok=True,parents=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")
    dur = total_frames/fps if fps>0 else 0
    print(f"Duration: {dur}")
    frame_interval = max(1,int(config.FRAME_INTERVAL_SECONDS*fps))
    print(f"Frame interval: {frame_interval}")

    extracted = []
    frame_ct=0

    while True:
        ret,frame = cap.read()

        if not ret:
            break

        if frame_ct%frame_interval == 0:
            timestamp = frame_ct/fps
            frame_filename = f"{frame_ct:06d}.jpg"
            frame_path = config.FRAMES_DIR / frame_filename

            print(f"Frame {frame_ct} at {timestamp} : {frame_filename}")

            cv2.imwrite(str(frame_path),frame,[int(cv2.IMWRITE_JPEG_QUALITY)    ,config.JPEG_QUALITY])

            extracted.append({
                "frame_path":str(frame_path),
                "timestamp":timestamp,
                "frame_ct":frame_ct
            })
            
        frame_ct+=1

    cap.release()
    print(f"Extracted {len(extracted)} frames from {video_path}") 
    return extracted
            
#downloaded_file = download_video('https://youtu.be/AEJNre8p0uY?si=wJFt9CZI2Z5EsV9B',config.VIDEO_DIR)

extracted = extract_video_frame(Path("videos/AEJNre8p0uY.mp4"))

print(extracted[:5])

            

