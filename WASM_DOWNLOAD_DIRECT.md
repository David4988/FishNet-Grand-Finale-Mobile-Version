# Direct WASM Download (No Build Required)

Instead of building from source, download pre-built WASM files directly.

## Option 1: Use Prebuilt Model Library

Visit the MLC-AI HuggingFace model library and download:
https://huggingface.co/mlc-ai/webllm-model-lib

Files needed:
- `phi-3-mini-128k-instruct-q4f16_1-MLC-webllm.wasm`

Download and save to: `public/models/phi-3-mini-128k-instruct-q4f16_1-MLC/`

## Option 2: Alternative - Use Prebuilt Model from CDN

Use this approach in your code instead (no local WASM needed):

```typescript
// In src/services/webllm.ts
import * as webllm from "@mlc-ai/web-llm";

const engine = await webllm.CreateMLCEngine("Phi-3-mini-128k-instruct-q4f16_1-MLC", {
  initProgressCallback: (progress) => console.log(progress),
});
```

This will automatically download the WASM from CDN instead of using local files.

## Quick Setup Without WSL Build

If you don't want to build WASM:
1. Remove the local WASM path requirement
2. Let WebLLM download WASM from CDN
3. Cache will store it locally

This is faster and requires no build process!
