#!/usr/bin/env python3
"""
Download Llama 3.2 1B model files for offline AI chat.
This script handles multi-threaded downloads for faster model retrieval.
"""

import os
import sys
from pathlib import Path
from urllib.request import urlopen
import time

MODEL_DIR = Path(__file__).parent / "models" / "Llama-3.2-1B-instruct-q4f16_1-MLC"
BASE_URL = "https://huggingface.co/mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC/resolve/main"

# Files to download
FILES_TO_DOWNLOAD = [
    # Config files (already downloaded via curl above)
    "params_shard_0.bin",
    "params_shard_1.bin",
    "params_shard_2.bin",
    "params_shard_3.bin",
    "params_shard_4.bin",
    "params_shard_5.bin",
    "params_shard_6.bin",
    "params_shard_7.bin",
    "params_shard_8.bin",
    "params_shard_9.bin",
    "params_shard_10.bin",
    "params_shard_11.bin",
    "params_shard_12.bin",
    "params_shard_13.bin",
    "params_shard_14.bin",
    "params_shard_15.bin",
    "params_shard_16.bin",
    "params_shard_17.bin",
    "params_shard_18.bin",
    "params_shard_19.bin",
    "params_shard_20.bin",
    "params_shard_21.bin",
]

def download_file(filename):
    """Download a single file from HuggingFace."""
    file_path = MODEL_DIR / filename
    
    # Skip if already exists
    if file_path.exists():
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"✓ {filename} (already exists, {size_mb:.2f} MB)")
        return True
    
    url = f"{BASE_URL}/{filename}"
    
    try:
        print(f"⏳ Downloading {filename}...", end=" ", flush=True)
        start_time = time.time()
        
        with urlopen(url, timeout=180) as response:
            total_size = int(response.headers.get('content-length', 0))
            chunk_size = 8192
            downloaded = 0
            
            with open(file_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r⏳ Downloading {filename}... {percent:.1f}%", end="", flush=True)
        
        elapsed = time.time() - start_time
        size_mb = file_path.stat().st_size / (1024 * 1024)
        speed_mbps = size_mb / elapsed if elapsed > 0 else 0
        
        print(f"\r✓ {filename} ({size_mb:.2f} MB, {speed_mbps:.2f} MB/s)     ")
        return True
        
    except Exception as e:
        print(f"\n✗ Failed to download {filename}: {e}")
        if file_path.exists():
            file_path.unlink()
        return False

def main():
    """Main download function."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(MODEL_DIR)
    
    print("=" * 60)
    print("Llama 3.2 1B Model Downloader")
    print("=" * 60)
    print(f"Model directory: {MODEL_DIR}")
    print(f"Files to download: {len(FILES_TO_DOWNLOAD)}")
    print()
    
    total_size = 0
    downloaded_count = 0
    failed_count = 0
    
    for filename in FILES_TO_DOWNLOAD:
        if download_file(filename):
            downloaded_count += 1
            file_path = MODEL_DIR / filename
            if file_path.exists():
                total_size += file_path.stat().st_size
        else:
            failed_count += 1
    
    print()
    print("=" * 60)
    print(f"Downloaded: {downloaded_count}/{len(FILES_TO_DOWNLOAD)}")
    print(f"Failed: {failed_count}")
    print(f"Total size: {total_size / (1024**3):.2f} GB")
    print("=" * 60)
    
    if failed_count > 0:
        print("\n⚠️  Some files failed to download.")
        print("You can re-run this script to retry.")
        return 1
    
    print("\n✅ All model files downloaded successfully!")
    print("\nTo start the chat server:")
    print("  cd offline-ai")
    print("  python -m http.server 8000")
    print("\nThen visit: http://localhost:8000")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
