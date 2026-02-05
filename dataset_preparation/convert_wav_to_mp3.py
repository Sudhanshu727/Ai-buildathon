#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from moviepy.editor import AudioFileClip
from tqdm import tqdm
from dotenv import load_dotenv

# Load env vars
load_dotenv(dotenv_path="../config/.env")
DATASET_PATH = os.getenv("DATASET_PATH", "E:/hackathon_dataset_v2")
CONVERTED_DATASET_PATH = os.getenv("CONVERTED_DATASET_PATH", "E:/ai_voice_detection_api/converted_dataset")

def convert_wav_to_mp3(wav_path, mp3_path):
    try:
        # Load and write using moviepy (automatic ffmpeg handling)
        with AudioFileClip(str(wav_path)) as audio:
            audio.write_audiofile(
                str(mp3_path),
                bitrate="192k",
                verbose=False,
                logger=None  # Silences the moviepy console output
            )
        return True
    except Exception as e:
        print(f"\nError converting {wav_path}: {e}")
        return False

def main():
    dataset_path = Path(DATASET_PATH)
    converted_path = Path(CONVERTED_DATASET_PATH)
    
    # scan for wavs
    wav_files = list(dataset_path.rglob("*.wav"))
    print(f"Found {len(wav_files)} files.")

    success_count = 0
    
    for wav_file in tqdm(wav_files, desc="Converting"):
        # Calculate output path maintaining structure
        relative_path = wav_file.relative_to(dataset_path)
        output_file = converted_path / relative_path.with_suffix(".mp3")
        
        # Ensure dir exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if output_file.exists():
            success_count += 1
            continue
            
        if convert_wav_to_mp3(wav_file, output_file):
            success_count += 1

    print(f"Done. Converted {success_count}/{len(wav_files)}")

if __name__ == "__main__":
    main()