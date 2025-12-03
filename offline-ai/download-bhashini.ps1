# Download Bhashini-Indic-LLM-1.5B model for MLC/WebLLM
# This script downloads model files from Hugging Face and saves to the correct directory

$modelDir = "d:\new\FISHnetsih\offline-ai\models\Bhashini-Indic-LLM-1.5B"
$huggingFaceBase = "https://huggingface.co/ai4bharat/Bhashini-Indic-LLM-1.5B/resolve/main"

# Create model directory if it doesn't exist
if (-not (Test-Path $modelDir)) {
    Write-Host "Creating directory: $modelDir" -ForegroundColor Green
    New-Item -ItemType Directory -Path $modelDir -Force | Out-Null
} else {
    Write-Host "Directory exists: $modelDir" -ForegroundColor Green
}

# List of essential files to download (starting with config files)
$files = @(
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors"
)

$ProgressPreference = 'SilentlyContinue'
$totalFiles = $files.Count
$downloadedCount = 0
$failedFiles = @()

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Bhashini-Indic-LLM-1.5B Model Download" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

foreach ($file in $files) {
    $downloadedCount++
    $url = "$huggingFaceBase/$file"
    $outputPath = Join-Path $modelDir $file
    
    Write-Host "[$downloadedCount/$totalFiles] Downloading: $file" -ForegroundColor Yellow
    Write-Host "  URL: $url" -ForegroundColor Gray
    Write-Host "  To: $outputPath" -ForegroundColor Gray
    
    try {
        # Check if file already exists
        if (Test-Path $outputPath) {
            $existingSize = (Get-Item $outputPath).Length
            Write-Host "  ✓ File already exists ($('{0:N0}' -f ($existingSize / 1MB)) MB)" -ForegroundColor Green
            continue
        }
        
        # Download file
        Invoke-WebRequest -Uri $url -OutFile $outputPath -TimeoutSec 300
        $fileSize = (Get-Item $outputPath).Length
        Write-Host "  ✓ Downloaded successfully ($('{0:N0}' -f ($fileSize / 1MB)) MB)" -ForegroundColor Green
    }
    catch {
        Write-Host "  ✗ Failed to download: $($_.Exception.Message)" -ForegroundColor Red
        $failedFiles += $file
    }
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Download Summary" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$successCount = $totalFiles - $failedFiles.Count
Write-Host "Total files: $totalFiles" -ForegroundColor White
Write-Host "Downloaded: $successCount" -ForegroundColor Green
Write-Host "Failed: $($failedFiles.Count)" -ForegroundColor $(if ($failedFiles.Count -gt 0) { "Red" } else { "Green" })

if ($failedFiles.Count -gt 0) {
    Write-Host "`nFailed files:" -ForegroundColor Red
    $failedFiles | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
}

Write-Host "`nModel saved to: $modelDir`n" -ForegroundColor Cyan

# Display downloaded files
Write-Host "Files in directory:" -ForegroundColor Yellow
Get-ChildItem $modelDir -File | Select-Object Name, @{N='SizeMB';E={[math]::Round($_.Length/1MB,2)}} | Format-Table -AutoSize

Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Convert model for MLC/WebLLM (if needed)" -ForegroundColor White
Write-Host "2. Update chat.html to use the new model" -ForegroundColor White
Write-Host "3. Test in offline-ai/chat.html" -ForegroundColor White
