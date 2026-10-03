$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$modelsDir = Join-Path $repoRoot "models"
$curl = Get-Command "curl.exe" -ErrorAction SilentlyContinue

if (-not $curl) {
    throw "curl.exe was not found. Install curl or use scripts/download_models.sh from WSL/Git Bash."
}

$directories = @("brain", "whisper", "piper", "minilm", "hf_cache")
foreach ($directory in $directories) {
    New-Item -ItemType Directory -Path (Join-Path $modelsDir $directory) -Force | Out-Null
}

function Download-Model {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    $partialPath = "$Destination.$([guid]::NewGuid().ToString('N')).partial"

    if ((Test-Path -LiteralPath $Destination -PathType Leaf) -and
        (Get-Item -LiteralPath $Destination).Length -gt 0) {
        Write-Host "Already present: $([System.IO.Path]::GetFileName($Destination))"
        return
    }

    Write-Host "Downloading $([System.IO.Path]::GetFileName($Destination))..."

    try {
        & $curl.Source --fail --location --retry 3 --output $partialPath $Url
        if ($LASTEXITCODE -ne 0) {
            throw "curl.exe failed with exit code $LASTEXITCODE for $Url"
        }
        if (-not (Test-Path -LiteralPath $partialPath -PathType Leaf)) {
            throw "curl.exe did not create the expected download: $partialPath"
        }
        if ((Get-Item -LiteralPath $partialPath).Length -eq 0) {
            throw "The downloaded file is empty: $Url"
        }

        Move-Item -LiteralPath $partialPath -Destination $Destination -Force
    }
    catch {
        if (Test-Path -LiteralPath $partialPath -PathType Leaf) {
            Remove-Item -LiteralPath $partialPath -Force
        }
        throw
    }
}

Write-Host "==> The Brain: Qwen 2.5 3B Instruct (Q4_K_M GGUF, ~2.2GB)"
Download-Model `
    "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf" `
    (Join-Path $modelsDir "brain\qwen2.5-3b-instruct-q4_k_m.gguf")

Write-Host "==> The Ears: whisper.cpp base.en model (~150MB)"
Download-Model `
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin" `
    (Join-Path $modelsDir "whisper\ggml-base.en.bin")

Write-Host "==> The Mouth: Piper voice en_US-lessac-medium (~60MB)"
Download-Model `
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx" `
    (Join-Path $modelsDir "piper\en_US-lessac-medium.onnx")
Download-Model `
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" `
    (Join-Path $modelsDir "piper\en_US-lessac-medium.onnx.json")

Write-Host "==> MiniLM and faster-whisper assets should be prefetched into models/minilm and models/whisper."
Write-Host "    Use the project's pinned model download tooling or copy complete local model directories there."

Write-Host "==> Done. Models placed under $modelsDir"
Write-Host "    Swap qwen2.5-3b-instruct-q4_k_m.gguf for a Llama 3.2 3B GGUF if you"
Write-Host "    prefer that brain - just update NOVA_LLM_MODEL_PATH in nova/config.py"
Write-Host "    or the corresponding environment variable."
