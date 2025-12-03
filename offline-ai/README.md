# Offline AI Chat - Llama 3.2 1B

A fully offline WebLLM chatbot running Llama 3.2 1B model entirely in your browser.

## Features

- ✅ **100% Offline** - No internet required after initial setup
- ✅ **Local Model** - Llama 3.2 1B (3.2 billion parameters, quantized to 4-bit)
- ✅ **Browser-based** - Uses WebLLM for in-browser inference
- ✅ **Fast Response** - GPU acceleration when available, falls back to WebAssembly
- ✅ **Lightweight** - Optimized model size (~2GB)
- ✅ **Privacy** - All data stays on your device

## Setup

### 1. Download Model Files

The model files (sharded parameters, tokenizer, config) should be downloaded to:
```
offline-ai/models/Llama-3.2-1B-instruct-q4f16_1-MLC/
```

Total size: ~2GB

Required files:
- `mlc-chat-config.json` - Model configuration
- `tokenizer.json` - Tokenizer vocabulary
- `tokenizer_config.json` - Tokenizer settings
- `params_shard_0.bin` through `params_shard_21.bin` - Model parameters (22 files)
- `ndarray-cache.json` - Weight layout
- `tensor-cache.json` - Tensor cache metadata

### 2. Serve the Application

You have several options:

**Option A: Python (Recommended)**
```bash
python -m http.server 8000
# Visit: http://localhost:8000
```

**Option B: Node.js (if http-server installed)**
```bash
npx http-server
```

**Option C: VS Code Live Server**
- Install "Live Server" extension
- Right-click `index.html` → "Open with Live Server"

**Option D: Any HTTP Server**
- Use your preferred static file server on port 8000+

### 3. Access the Application

Navigate to:
```
http://localhost:8000
```

## First Run

**Initial load:** 30-60 seconds
- Model is loaded into browser memory
- Subsequent messages are much faster (~2-5 seconds per response)

## Usage

1. Type your message in the text area
2. Press Enter or click Send
3. Wait for the AI response (no internet needed!)

## System Requirements

- **Browser:** Chrome, Edge, or Firefox (WebGPU support recommended)
- **RAM:** 8GB+ recommended (model uses 3-4GB)
- **Disk:** 2.5GB for model files
- **Storage:** IndexedDB or browser cache for model weights

## Performance

- **First response:** 30-60 seconds (model loading)
- **Subsequent responses:** 2-5 seconds (CPU) / 1-2 seconds (GPU)
- **Tokens/second:** 2-5 on CPU, 10+ on GPU

## Troubleshooting

### Model fails to load
- Check that all model files are in `models/Llama-3.2-1B-instruct-q4f16_1-MLC/`
- Verify file names are exact (case-sensitive)
- Check browser console for detailed errors

### Slow responses
- WebGPU not available - using WebAssembly fallback
- System may be under-resourced
- Try restarting browser to free memory

### "Entry not found" errors
- Model files are incomplete or corrupt
- Re-download from HuggingFace repository
- Clear browser cache and try again

### CORS errors
- Must be served over HTTP/HTTPS (not file://)
- Use one of the recommended server options above

## Model Information

**Model:** Llama-3.2-1B-Instruct (4-bit quantized)
- Parameters: 1.2B (quantized from original)
- Format: WebLLM compatible ONNX/WASM
- Quantization: q4f16_1 (4-bit weights)
- Source: [HuggingFace mlc-ai](https://huggingface.co/mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC)

## Architecture

```
offline-ai/
├── index.html          # Chat UI
├── main.js             # WebLLM engine & chat logic
├── package.json        # Dependencies
├── README.md           # This file
└── models/
    └── Llama-3.2-1B-instruct-q4f16_1-MLC/
        ├── mlc-chat-config.json
        ├── tokenizer.json
        ├── tokenizer_config.json
        ├── ndarray-cache.json
        ├── tensor-cache.json
        ├── params_shard_0.bin
        ├── params_shard_1.bin
        ├── ...
        └── params_shard_21.bin
```

## Limitations

- Responses limited to 512 tokens
- Context window: ~2048 tokens (model limitation)
- No multi-turn conversation memory yet
- Single user session

## Privacy

✅ **No data sent to servers**
- All inference happens locally
- No telemetry
- No tracking
- Complete privacy

## License

Model weights from Meta Llama (License Agreement)
WebLLM from MLC-AI (Apache 2.0)
UI created for FISHnetsih project

---

**Note:** First-time setup requires ~2.5GB disk space and ~4GB RAM. Subsequent runs are much faster.
