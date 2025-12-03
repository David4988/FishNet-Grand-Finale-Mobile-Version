@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0models\Llama-3.2-1B-instruct-q4f16_1-MLC"

echo Starting model file download...
echo.

REM Download files sequentially with better error handling
for /L %%i in (0,1,21) do (
    set "shard=params_shard_%%i.bin"
    if not exist "!shard!" (
        echo Downloading !shard!...
        powershell -NoProfile -Command ^
            "$url = 'https://huggingface.co/mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC/resolve/main/!shard!'; " ^
            "$file = '!shard!'; " ^
            "try { " ^
            "  $response = Invoke-WebRequest -Uri $url -OutFile $file -PassThru -TimeoutSec 180 -ErrorAction Stop; " ^
            "  $size = (Get-Item $file).Length; " ^
            "  Write-Host \"✓ Downloaded !shard! ($([Math]::Round($size/1MB, 2)) MB)\"; " ^
            "} catch { " ^
            "  Write-Host \"✗ Failed: !shard! - $_\"; " ^
            "}"
    ) else (
        echo Skipped !shard! (already exists)
    )
)

echo.
echo Download complete!
pause
