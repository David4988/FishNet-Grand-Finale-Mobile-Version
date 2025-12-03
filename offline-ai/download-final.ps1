$ModelDir = "d:\new\FISHnetsih\offline-ai\models\Llama-3.2-1B-instruct-q4f16_1-MLC"
$BaseUrl = "https://huggingface.co/mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC/resolve/main"
$MaxRetries = 3

function Download-File {
    param(
        [string]$Filename,
        [int]$Retry = 0
    )
    
    $FilePath = Join-Path $ModelDir $Filename
    
    if (Test-Path $FilePath) {
        $Size = (Get-Item $FilePath).Length
        $SizeMB = [math]::Round($Size / 1048576, 2)
        Write-Host "OK $Filename ($SizeMB MB)"
        return $true
    }
    
    $Url = "$BaseUrl/$Filename"
    
    try {
        Write-Host "Downloading $Filename..." -NoNewline
        $ProgressPreference = 'SilentlyContinue'
        
        Invoke-WebRequest -Uri $Url -OutFile $FilePath -TimeoutSec 300 -UseBasicParsing -ErrorAction Stop
        
        $Size = (Get-Item $FilePath).Length
        $SizeMB = [math]::Round($Size / 1048576, 2)
        Write-Host " OK ($SizeMB MB)"
        return $true
    }
    catch {
        if ($Retry -lt $MaxRetries) {
            Write-Host " Retry $($Retry + 1)/$MaxRetries"
            Start-Sleep -Seconds 5
            return Download-File -Filename $Filename -Retry ($Retry + 1)
        }
        else {
            Write-Host " FAILED"
            if (Test-Path $FilePath) { Remove-Item $FilePath }
            return $false
        }
    }
}

Write-Host "Downloading Llama 3.2 1B model files..."
Write-Host ""

$Success = 0
for ($i = 0; $i -le 21; $i++) {
    $Shard = "params_shard_$i.bin"
    if (Download-File -Filename $Shard) {
        $Success++
    }
}

$Total = (Get-ChildItem $ModelDir -File | Measure-Object).Count
Write-Host ""
Write-Host "Downloaded: $Total/27 files"
Write-Host "Completed: $Success/22 shards"

if ($Total -eq 27) {
    Write-Host ""
    Write-Host "SUCCESS! All model files downloaded."
    Write-Host ""
    Write-Host "Next: Start the server:"
    Write-Host "  cd d:\new\FISHnetsih\offline-ai"
    Write-Host "  python -m http.server 8000"
    Write-Host ""
    Write-Host "Then visit: http://localhost:8000"
}
else {
    Write-Host ""
    Write-Host "Remaining: $($27 - $Total) files"
}
