# Setup Offline WebLLM Model for FishNet

## Overview
This guide walks you through building and configuring the WebLLM WASM runtime for the **Phi-3-mini-128k-instruct-q4f16_1-MLC** model to enable fully offline AI chat in your FishNet application.

## Current Status
✅ **Already Completed:**
- Phi-3-mini model weights downloaded (all `params_shard_*.bin` files)
- Tokenizer files downloaded (`tokenizer.json`, `tokenizer_config.json`)
- Model config downloaded (`mlc-chat-config.json`)
- WebLLM service configured and ready
- App configuration updated to use Phi-3-mini

❌ **Still Needed:**
- WASM runtime file (`phi-3-mini-128k-instruct-q4f16_1-MLC-webllm.wasm`)

## Step 1: Build WASM Runtime

### Prerequisites
- WSL (Windows Subsystem for Linux) with Ubuntu installed
- At least 50GB free disk space (for build artifacts)
- 30-60 minutes for the build process

### Instructions

1. **Copy the build script to WSL:**
   - Copy `build_webllm_phi3.sh` from your Windows workspace to your WSL Ubuntu home directory:
     ```powershell
     # From Windows PowerShell
     Copy-Item "d:\new\FISHnetsih\build_webllm_phi3.sh" -Destination "\\wsl$\Ubuntu\home\<your-username>\build_webllm_phi3.sh"
     ```

2. **Make the script executable:**
   ```bash
   chmod +x build_webllm_phi3.sh
   ```

3. **Run the build script:**
   ```bash
   ./build_webllm_phi3.sh
   ```

   This will:
   - Install all required dependencies (build-essential, cmake, llvm-15, etc.)
   - Install Emscripten (emsdk) for WASM compilation
   - Clone the MLC-LLM and WebLLM repositories
   - Configure TVM for WebGPU support
   - Build the TVM runtime
   - Compile your Phi-3-mini model
   - Generate the WASM runtime file

4. **Wait for completion:**
   - The build typically takes 30-60 minutes depending on your system
   - Monitor the output for any errors
   - At the end, you should see:
     ```
     ✓ WASM file successfully created!
     Final model folder contents:
     -rw-r--r-- ... params_shard_0.bin
     -rw-r--r-- ... params_shard_1.bin
     ...
     -rw-r--r-- ... phi-3-mini-128k-instruct-q4f16_1-MLC-webllm.wasm
     ```

## Step 2: Copy WASM File to Your Project

After the build completes, the WASM file needs to be in your project:

```bash
# From WSL, copy the WASM file back to your Windows project
cp ~/web-llm/public/models/phi-3-mini-128k-instruct-q4f16_1-MLC/phi-3-mini-128k-instruct-q4f16_1-MLC-webllm.wasm /mnt/d/new/FISHnetsih/public/models/phi-3-mini-128k-instruct-q4f16_1-MLC/
```

## Step 3: Verify Installation

Check that all files are present in `public/models/phi-3-mini-128k-instruct-q4f16_1-MLC/`:

```powershell
# From Windows PowerShell
Get-ChildItem d:\new\FISHnetsih\public\models\phi-3-mini-128k-instruct-q4f16_1-MLC\ | Select-Object Name, Length
```

You should see:
- `params_shard_0.bin` (49.3 MB)
- `params_shard_1.bin` (25.2 MB)
- ... (all other shards)
- `params_shard_48.bin` (21.2 MB)
- `mlc-chat-config.json` (2.13 kB)
- `tokenizer.json`
- `tokenizer_config.json`
- `phi-3-mini-128k-instruct-q4f16_1-MLC-webllm.wasm` (~200-300 MB)

## Step 4: Build and Test Your App

Once all files are in place:

```powershell
# From your project directory
npm run dev
```

Then:
1. Open the chat interface in your app
2. Click "Start Chat" or send a message
3. The model should load (this takes 1-2 minutes on first load)
4. Test the chat functionality offline

## Troubleshooting

### Build Fails with "emsdk not found"
- Make sure you're running the script inside WSL (Ubuntu)
- Verify Emscripten is installed: `./emsdk/emsdk --version`

### Build Takes Too Long
- This is normal. The build includes compiling TVM, the model, and WASM runtime
- Typical time: 30-60 minutes

### WASM File Not Generated
- Check the build output for errors
- Ensure you have at least 50GB free disk space
- Try running the script again or manually following the steps

### Model Doesn't Load in App
- Verify all files exist in `public/models/phi-3-mini-128k-instruct-q4f16_1-MLC/`
- Check browser console for errors (F12 → Console tab)
- Ensure service worker is registered: Check `/public/sw.js`

## Next Steps

After successful setup:
1. Test offline functionality by disconnecting from the internet
2. Deploy to Replit or your server
3. The app will work fully offline with all model inference running locally

## Additional Resources

- [WebLLM Documentation](https://webllm.mlc.ai/)
- [MLC-LLM GitHub](https://github.com/mlc-ai/mlc-llm)
- [Phi-3 Model Info](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)

## Support

If you encounter issues:
1. Check the error message carefully
2. Review the build script output
3. Verify all dependencies are installed
4. Check system disk space and RAM

---

**Estimated Total Time:** 30-60 minutes for build + 5 minutes for verification + testing
