import os
import random
import shutil
from pathlib import Path

# Define paths
LIBRISPEECH_DIR = r"dataset\LibriSpeech"
INSTANT_DIR = r"dataset\instant"
MAX_FILES = 100

def get_audio_files_from_librispeech(root_dir):
    """
    Recursively find all .flac files in LibriSpeech directory structure.
    Returns list of full file paths.
    """
    audio_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.flac'):
                audio_files.append(os.path.join(root, file))
    return audio_files

def main():
    # Create instant directory if it doesn't exist
    Path(INSTANT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Get all audio files
    print(f"Searching for audio files in {LIBRISPEECH_DIR}...")
    all_files = get_audio_files_from_librispeech(LIBRISPEECH_DIR)
    
    if not all_files:
        print(f"No audio files found in {LIBRISPEECH_DIR}")
        return
    
    print(f"Found {len(all_files)} total audio files")
    
    # Randomly sample up to MAX_FILES
    sample_size = min(MAX_FILES, len(all_files))
    sampled_files = random.sample(all_files, sample_size)
    
    print(f"Sampling {sample_size} random files...")
    
    # Copy sampled files to instant folder
    for idx, src_file in enumerate(sampled_files, 1):
        filename = os.path.basename(src_file)
        dst_file = os.path.join(INSTANT_DIR, filename)
        
        try:
            shutil.copy2(src_file, dst_file)
            print(f"[{idx}/{sample_size}] Copied: {filename}")
        except Exception as e:
            print(f"Error copying {filename}: {e}")
    
    print(f"\nCompleted! {sample_size} files copied to {INSTANT_DIR}")

if __name__ == "__main__":
    main()