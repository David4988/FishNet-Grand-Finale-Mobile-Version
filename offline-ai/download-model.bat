@echo off
REM Download Llama 3.2 1B Model for Offline AI Chat
REM This script downloads the remaining model files

setlocal enabledelayedexpansion
cd /d "%~dp0models\Llama-3.2-1B-instruct-q4f16_1-MLC" || (
    echo Error: Model directory not found
    exit /b 1
)

echo Downloading remaining Llama 3.2 1B model files...
echo.

set "baseUrl=https://huggingface.co/mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC/resolve/main/"

REM Download parameter shards
for /L %%i in (0,1,21) do (
    set "file=params_shard_%%i.bin"
    if not exist "!file!" (
        echo Downloading !file!...
        powershell -Command "Invoke-WebRequest -Uri '%baseUrl%!file!' -OutFile '!file!' -TimeoutSec 120"
        if errorlevel 1 (
            echo Failed to download !file!
        ) else (
            for %%A in ("!file!") do set "size=%%~zA"
            echo   Downloaded: !size! bytes
        )
    ) else (
        echo Already exists: !file!
    )
)

echo.
echo Download complete!
echo Model files: %baseUrl%
