# Offline AI Setup Instructions

## Quick Start

### 1. Model Files Are Downloading
The Llama 3.2 1B model files (~550MB total) are being downloaded to:
```
offline-ai/models/Llama-3.2-1B-instruct-q4f16_1-MLC/
```

This may take 5-15 minutes depending on your internet speed.

### 2. Check Download Status
Run this in PowerShell to monitor progress:
```powershell
Get-ChildItem "offline-ai\models\Llama-3.2-1B-instruct-q4f16_1-MLC" -File | 
  Measure-Object | Select-Object -ExpandProperty Count
```

You should see: **26 files** when complete
- 1x mlc-chat-config.json
- 1x tokenizer.json
- 1x tokenizer_config.json
- 1x ndarray-cache.json
- 1x tensor-cache.json
- 22x params_shard_*.bin (0-21)

### 3. Start the Server
Once downloads complete, run:

**Option A: Python (Recommended)**
```powershell
cd offline-ai
python -m http.server 8000
```

**Option B: Using Node.js**
```powershell
cd offline-ai
npx http-server
```

### 4. Open in Browser
Navigate to:
```
http://localhost:8000
```

First load: 30-60 seconds (model initialization)
Subsequent messages: 2-5 seconds each

## What's Included

- **index.html** - Modern chat UI with real-time typing indicators
- **main.js** - WebLLM engine that runs Llama 3.2 1B offline
- **models/** - Local model files (not synced to main project)

## Features

✅ 100% Offline - No internet after first download
✅ Real AI - Uses actual Llama 3.2 1B model (not templates)
✅ Private - All responses computed locally
✅ Fast - GPU acceleration when available

## Troubleshooting

### "Failed to initialize model"
- Check that model files are fully downloaded (26 files total)
- Clear browser cache and try again
- Ensure you're accessing via http://localhost (not file://)

### "CORS error" or "Entry not found"
- Model must be served over HTTP
- Don't open index.html directly - use Python/Node server

### Slow responses on first message
- Normal! Model is loading (~4GB into memory)
- Wait 30-60 seconds for first response
- Subsequent messages are much faster

### Model files still downloading
- Download speed depends on internet connection
- Can be 5-15 minutes for full model
- Keep terminal window open

## File Structure

```
offline-ai/
├── index.html                  (Chat interface)
├── main.js                     (AI engine)
├── package.json               (Dependencies)
├── README.md                  (Full documentation)
├── SETUP.md                   (This file)
└── models/
    └── Llama-3.2-1B-instruct-q4f16_1-MLC/
        ├── mlc-chat-config.json
        ├── tokenizer.json
        ├── tokenizer_config.json
        ├── ndarray-cache.json
        ├── tensor-cache.json
        ├── params_shard_0.bin through params_shard_21.bin
```

## Next Steps

1. Wait for downloads to complete
2. Start HTTP server
3. Open http://localhost:8000
4. Chat with Llama 3.2 1B!

---

**Questions?** Check README.md for detailed documentation.
