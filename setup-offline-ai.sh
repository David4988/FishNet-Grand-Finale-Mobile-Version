#!/bin/bash
# OFFLINE AI SETUP QUICK START SCRIPT
# This script helps you download and set up WebLLM models for offline use

set -e

echo "======================================"
echo "Fish Net - Offline AI Setup"
echo "======================================"
echo ""

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_LLM_SRC="$PROJECT_ROOT/attached_assets/web-llm"
MODELS_DIR="$PROJECT_ROOT/models"

# Create models directory
echo "[1/5] Creating models directory..."
mkdir -p "$MODELS_DIR"

# Option to download a model (requires curl and ~1-3 GB space)
echo ""
echo "[2/5] Model download instructions:"
echo ""
echo "Choose a model and download it to $MODELS_DIR/"
echo ""
echo "Recommended models for browser:"
echo "  1. Phi-3 Mini (2.3B, fast, good quality)"
echo "     Source: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf"
echo "     File: phi-3-mini-4k-instruct-q4.gguf"
echo ""
echo "  2. TinyLlama (1.1B, very fast, smaller)"
echo "     Source: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
echo "     File: tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
echo ""
echo "  3. Mistral (7B, better quality, slower)"
echo "     Source: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
echo "     File: mistral-7b-instruct-v0.2.Q4_K_M.gguf"
echo ""
echo "Steps:"
echo "  1. Visit one of the sources above"
echo "  2. Download the GGUF file"
echo "  3. Extract to $MODELS_DIR/ if compressed"
echo "  4. Rename model file to match WebLLM expectations"
echo ""

# Check if WebLLM source exists
if [ ! -d "$WEB_LLM_SRC" ]; then
    echo "[ERROR] WebLLM not found at $WEB_LLM_SRC"
    echo "Please run: git clone https://github.com/mlc-ai/web-llm attached_assets/web-llm"
    exit 1
fi

echo "[3/5] WebLLM found at: $WEB_LLM_SRC"

# Instructions for WebLLM runtime
echo ""
echo "[4/5] WebLLM Runtime setup:"
echo "  The WebLLM npm package is already installed."
echo "  The browser bundle will be loaded automatically from node_modules"
echo "  or you can place a prebuilt bundle in /web-llm/"
echo ""

# Build the app
echo "[5/5] Building the application..."
cd "$PROJECT_ROOT"
npm run build

echo ""
echo "======================================"
echo "✅ Setup complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "  1. Download a model (see instructions above)"
echo "  2. Place model files in $MODELS_DIR/"
echo "  3. Run: npm run dev"
echo "  4. Visit http://localhost:5173"
echo "  5. Go to the AI Chat tab and wait for model to load"
echo ""
echo "For production deployment, see OFFLINE_AI_SETUP.md"
echo ""
