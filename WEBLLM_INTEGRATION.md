Steps to integrate WebLLM into this project (offline-capable PWA)

1) Clone the WebLLM repository

  git clone https://github.com/mlc-ai/web-llm attached_assets/web-llm

2) Move the chat example files into the chat view (optional locations shown)

  # POSIX (Linux / macOS / Replit shell)
  mv attached_assets/web-llm/examples/chat/* src/components/social/

  # PowerShell (Windows)
  Move-Item -Path attached_assets\web-llm\examples\chat\* -Destination src\components\social\

  After moving, ensure that `AIChat.tsx` (this project) is updated to use the web-llm script (we added runtime detection).

3) Move the models folder into the project root so the PWA can serve them from `/models`

  # POSIX
  mv attached_assets/web-llm/models ./models

  # PowerShell
  Move-Item -Path attached_assets\web-llm\models -Destination .\models

4) Remove unneeded folders from the cloned repo to keep the project clean (optional)

  # remove the rest of the cloned repo after extracting what you need
  rm -rf attached_assets/web-llm/.git
  rm -rf attached_assets/web-llm/.github
  # keep only the runtime JS and the models you need

5) Replit / static server configuration

  - This project contains a `.replit` file and a `package.json` script `start:replit` that runs:

    npx serve -s . --single --listen 0.0.0.0:3000

  - `serve` will expose the project root so `/models`, `/dist`, `/assets`, and `/js` are available.

6) Update `index.html` to load WebLLM scripts and initialize the model

  - This repo already contains a small dynamic bootstrap in `index.html` which tries common script locations:
    - `/web-llm/webllm.mjs`, `/web-llm/webllm.js`, `/js/webllm.mjs`, `/js/webllm.js`
  - Place the WebLLM bundle (the single-file browser script) in either `/web-llm/` or `/js/`.
  - Place the model files under `/models/` (for example `/models/small-model/` or `/models/model.tflite` depending on the WebLLM build).

7) Service worker & manifest

  - The service worker (`public/sw.js`) is already configured to:
    - Register at `/sw.js` (registered in `src/services/pwa.ts`)
    - Cache `/models/model.tflite`, `/models/species.json` and the new local news file `/news/local-news.json`.
  - If you add new model files, add them to the `STATIC_ASSETS` array in `public/sw.js` so they are pre-cached for offline use.

8) How the chat UI works now

  - `src/components/social/AIChat.tsx` will attempt to detect a WebLLM bundle exposed on `window` (common names: `WebLLM`, `webllm`, `WebLLMClient`, `WebLLMWorker`).
  - If a runtime is found, the component tries common methods (`generate`, `run`, `predict`) to request text completions.
  - If no WebLLM is present, the UI falls back to the existing simulated response generator (works fully offline).

9) Quick local test (development)

  # Install deps
  npm install

  # Run dev server (Vite)
  npm run dev

  # Or serve the built app for a production-like static server (useful for Replit)
  npm run build
  npm run start:replit

10) Notes and troubleshooting

  - News fetch: The app expects a `VITE_NEWS_API_KEY` environment variable for NewsAPI; if missing or blocked, the app will load `public/news/local-news.json` as a fallback.
  - CORS: If the external news API is blocked due to CORS on the target origin, prefer using the local news fallback in `public/news/local-news.json`, or run a simple proxy on a server you control.
  - WebLLM model size: Use the smallest model supported by the WebLLM build for best in-browser performance (e.g., a 7B or 3B smaller quantized flavor if available).

If you'd like, I can:
- Attempt to add a small (placeholder) WebLLM bundle in `/web-llm` and a tiny example model so you can test fully offline, or
- Walk through the exact commands to move the `web-llm/examples/chat` files into `src/components/social` and wire any missing imports.
