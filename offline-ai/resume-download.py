#!/usr/bin/env python3
"""
Resume Llama 3.2 1B model download with retry logic.
Continues from where the previous download left off.
"""

import os
from pathlib import Path
from urllib.request import urlopen, Request
import time

MODEL_DIR = Path(__file__).parent / "models" / "Llama-3.2-1B-instruct-q4f16_1-MLC"
BASE_URL = "https://huggingface.co/mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC/resolve/main"
MAX_RETRIES = 3

def download_file(filename, retry=0):
    """Download a single file with retry logic."""
    file_path = MODEL_DIR / filename
    
    if file_path.exists():
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"✓ {filename} ({size_mb:.2f} MB)")
        return True
    
    url = f"{BASE_URL}/{filename}"
    
    try:
        print(f"⏳ {filename}...", end=" ", flush=True)
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        
        with urlopen(req, timeout=300) as response:
            with open(file_path, 'wb') as f:
                f.write(response.read())
        
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"✓ ({size_mb:.2f} MB)")
        return True
        
    except Exception as e:
        if retry < MAX_RETRIES:
            print(f"⚠ Retry {retry + 1}/{MAX_RETRIES}")
            time.sleep(5)
            return download_file(filename, retry + 1)
        else:
            print(f"✗ Failed after {MAX_RETRIES} retries")
            if file_path.exists():
                file_path.unlink()
            return False

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(MODEL_DIR)
    
    # All 22 parameter shards
    shards = [f"params_shard_{i}.bin" for i in range(22)]
    
    print("Resuming model download...\n")
    
    successful = 0
    for shard in shards:
        if download_file(shard):
            successful += 1
    
    total_files = len(shards) + 5  # 22 shards + 5 config files
    downloaded = len([f for f in MODEL_DIR.iterdir() if f.is_file()])
    
    print(f"\n✓ Downloaded: {downloaded}/{total_files}")
    
    if downloaded == total_files:
        print("✅ Complete!")
    else:
        print(f"⚠ Remaining: {total_files - downloaded} files")

if __name__ == "__main__":
    main()
