# Offline AI Fishing Assistant with WebLLM

This guide explains how the offline LLM integration works in the Fish Net application and how to deploy it.

## Architecture

The application now includes a fully offline AI fishing assistant powered by **WebLLM**, which runs large language models directly in the browser using WebGPU/WebAssembly.

### Key Components

1. **WebLLM Service** (`src/services/webllm.ts`)
   - Manages the MLCEngine instance
   - Handles model initialization and inference
   - Builds context-aware prompts that include recent user catches
   - Supports multi-language responses based on app language settings

2. **AI Chat Component** (`src/components/social/AIChat.tsx`)
   - Displays chat interface with WebLLM responses
   - Shows initialization status and errors
   - Detects language setting and communicates with WebLLM in that language
   - Integrates with catch history from IndexedDB
   - Works completely offline after model download

3. **Service Worker** (`public/sw.js`)
   - Caches WebLLM runtime files and models
   - Enables offline functionality for the entire app

## Setup & Deployment

### Step 1: Extract WebLLM Files (Already Done)

The web-llm repository has been cloned to `attached_assets/web-llm/`. Now extract the necessary files:

#### Option A: Using the prebuild dist (Recommended)

If WebLLM provides a prebuilt bundle, place it in:
- `/web-llm/webllm.mjs` (or `.js` variant)
- `/js/webllm.mjs` (alternate location)

#### Option B: Build WebLLM locally

```bash
cd attached_assets/web-llm
npm install
npm run build
```

The built bundle will be in `lib/` directory. Copy the UMD or ESM bundle to `/web-llm/`.

### Step 2: Download/Place Models

WebLLM requires model files (GGUF or other compatible formats). Place them in `/models/`:

```
/models/
  ├── Phi-3-mini-4k-instruct-q4f32_1/
  │   ├── model.bin
  │   ├── tokenizer.json
  │   └── ...
  └── (other models)
```

**Model sources:**
- [Hugging Face](https://huggingface.co) - search for quantized Phi-3 or Mistral models
- [WebLLM Model Zoo](https://github.com/mlc-ai/web-llm) - see repo for official models
- [GGML Models](https://github.com/ggerganov/llama.cpp) - compatible quantized models

**Recommended models for browser**:
- `Phi-3-mini-4k-instruct-q4f32_1` (lightweight, fast)
- `Mistral-7B-Instruct-v0.3-q4` (larger, better quality)
- `TinyLlama-1.1B` (very small, suitable for low-end devices)

### Step 3: Build the Application

```bash
npm install
npm run build
```

This creates a production build in `dist/`.

### Step 4: Deploy to Replit (or Your Server)

#### For Replit:

1. Push the code to Replit:
   ```bash
   git push replit master
   ```

2. In Replit, the `.replit` file and `start:replit` script will automatically serve the app:
   ```bash
   npm run start:replit
   ```
   This runs `npx serve -s . --single --listen 0.0.0.0:3000`

3. The static server will serve:
   - `/dist/` - built app
   - `/models/` - LLM models
   - `/news/` - offline news
   - `/public/` - manifest, service worker, etc.

#### For Other Servers:

Deploy the `dist/` folder and ensure these directories are also served from root:
- `/models/` - place model files here
- `/news/` - local news JSON (already included)
- `/public/sw.js` - copy to root as `/sw.js`
- `/public/manifest.json` - copy to root

**Example Nginx config:**
```nginx
server {
    listen 3000;
    root /path/to/project;

    # Serve SPA routes
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Serve models with long cache headers
    location /models/ {
        add_header Cache-Control "public, max-age=31536000";
        try_files $uri =404;
    }

    # Service worker always fresh
    location /sw.js {
        add_header Cache-Control "no-cache, no-store, must-revalidate";
    }
}
```

## How It Works

### Initialization Flow

1. **App Loads** → `index.html` includes bootstrap script
2. **Bootstrap Script** tries to import WebLLM from common paths:
   - `/web-llm/webllm.mjs`
   - `/web-llm/webllm.js`
   - `/js/webllm.mjs`
   - `/js/webllm.js`
3. **AIChat Component** mounts and calls `webllmChatService.initialize()`
4. **WebLLM Engine** initializes and loads the model from `/models/`
5. **Model Download** happens on first use (cached in IndexedDB for future loads)
6. **Service Worker** registers and caches all static assets and models
7. **User sees** "Loading offline AI model..." until ready

### Chat Flow

1. **User sends message**
2. **Recent catches fetched** from IndexedDB
3. **System prompt built** with language-specific instructions + catch context
4. **WebLLM.generate()** called with user message + system prompt
5. **Model runs in browser** (no API call, fully offline)
6. **Response displayed** in chat UI

### Language Support

The AI responds in the user's selected language (from settings). Supported languages:
- English (en)
- Hindi (hi)
- Tamil (ta)
- Telugu (te)
- Kannada (kn)
- Malayalam (ml)
- Gujarati (gu)
- Marathi (mr)
- Bengali (bn)
- Punjabi (pa)
- Odia (or)
- Marwari (mwr)

System prompts are translated for each language and include context about fishing.

## Troubleshooting

### Model doesn't load

**Issue**: "Offline AI model initialization failed"

**Solutions**:
1. Check browser console for specific error
2. Verify `/models/` directory has model files
3. Ensure model name in `AIChat.tsx` matches actual model folder
4. Try smaller model: change `preferredModel` in `AIChat.tsx` line ~50
5. Check IndexedDB quota (DevTools → Application → Storage)

### App works online but offline chat fails

**Issue**: Chat works, but stops working after going offline

**Solutions**:
1. Verify service worker is registered: DevTools → Application → Service Workers
2. Check that `/sw.js` is served and has caching rules for `/models/`
3. Ensure first visit was with model fully loaded (so it cached)
4. Clear cache and revisit to force re-caching

### Model takes too long to load

**Issue**: First load takes minutes

**Solutions**:
1. This is normal for first load (model download + IndexedDB cache)
2. Subsequent loads should be faster (<30s)
3. Use smaller model (Phi-3 mini instead of 7B variant)
4. Check network speed (large models can be 2-8 GB)
5. Consider preloading on page idle time

### High memory usage / crashes on older devices

**Issue**: Chat slows down or crashes, especially on phones

**Solutions**:
1. Use smaller, quantized models (q4, q5 variants)
2. Reduce max_tokens in `webllm.ts` (line 107, change from 512 to 256)
3. Reduce temperature (line 106, change from 0.7 to 0.3)
4. Consider using TinyLlama-1.1B instead

## Environment Variables

Create a `.env` file if needed:

```env
VITE_NEWS_API_KEY=your_api_key_here  # For online news (optional)
```

News API is optional. The app falls back to `/news/local-news.json` if offline or key missing.

## File Structure

```
project/
├── src/
│   ├── services/
│   │   ├── webllm.ts (new - WebLLM service)
│   │   ├── database.ts (catches & history)
│   │   └── ...
│   ├── components/social/
│   │   └── AIChat.tsx (new - Chat UI)
│   └── ...
├── public/
│   ├── sw.js (service worker - updated)
│   ├── manifest.json
│   ├── news/
│   │   └── local-news.json (offline fallback)
│   └── ...
├── models/ (you create this)
│   └── Phi-3-mini-4k-instruct-q4f32_1/
│       ├── model.bin (or model.gguf)
│       ├── tokenizer.json
│       └── ...
├── web-llm/ (you extract this)
│   ├── webllm.mjs (or .js)
│   └── ...
├── dist/ (built app - generated by `npm run build`)
└── ...
```

## Performance Tips

1. **Reduce Model Size**: Use 4-bit quantized models (q4) instead of full precision
2. **Batch Requests**: Let the model finish before sending next message
3. **Clear Cache**: If IndexedDB grows too large, use DevTools → Storage → Clear all
4. **Use WebGPU**: Ensure user's browser supports WebGPU for acceleration (Chrome/Edge preferred)
5. **Preload Models**: Add preload hint in `index.html` for model files

## Security & Privacy

✅ **All processing happens in the browser** - no data sent to external servers
✅ **Catches data never leaves IndexedDB** - all local storage
✅ **Models cached locally** - no repeated downloads
✅ **Service worker caches offline** - works fully without internet

**Note**: On first load, the model file downloads from the source URL. After that, it's cached locally.

## Testing Offline

1. **In DevTools:**
   - Open Network tab
   - Go to Offline mode (DevTools → Network → Offline)
   - Load the app and try the chat

2. **In PWA:**
   - Install the app from browser install prompt
   - Quit browser completely
   - Open installed app
   - Chat should work offline

## Next Steps

1. Download or build the WebLLM bundle and place in `/web-llm/`
2. Download a small quantized model (e.g., Phi-3 mini) and place in `/models/`
3. Run `npm run build`
4. Deploy to Replit or your server
5. Test online first, then go offline and test chat

## Support & Resources

- [WebLLM Documentation](https://mlc.ai/web-llm)
- [MLCEngine API](https://github.com/mlc-ai/web-llm/tree/main/src)
- [Hugging Face Model Hub](https://huggingface.co)
- [Browser Support Matrix](https://github.com/mlc-ai/web-llm#platform-support)

---

**Status**: ✅ Fully integrated and ready for model files to be added.
