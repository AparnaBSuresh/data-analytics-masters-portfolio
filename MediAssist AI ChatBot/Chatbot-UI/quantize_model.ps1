# PowerShell script to quantize F16 GGUF model to Q4_K_M
# Run this after the quantize tool is built

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Model Quantization Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Add CMake to PATH if not already there
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    $env:PATH += ";C:\Program Files\CMake\bin"
    Write-Host "Added CMake to PATH" -ForegroundColor Green
}

# Check if quantize tool exists (it's called llama-quantize.exe on Windows)
$quantizePath = "llama.cpp\build\bin\Release\llama-quantize.exe"
if (-not (Test-Path $quantizePath)) {
    Write-Host "ERROR: Quantize tool not found at: $quantizePath" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please build it first:" -ForegroundColor Yellow
    Write-Host "  cd llama.cpp\build" -ForegroundColor White
    Write-Host "  `$env:PATH += ';C:\Program Files\CMake\bin'" -ForegroundColor White
    Write-Host "  cmake .. -DLLAMA_CURL=OFF" -ForegroundColor White
    Write-Host "  cmake --build . --config Release --target llama-quantize" -ForegroundColor White
    Write-Host ""
    exit 1
}

Write-Host "Quantize tool found!" -ForegroundColor Green
Write-Host ""

# Check if F16 model exists
$f16Path = "models\medllama-3b-gguf\medllama-3b-f16.gguf"
if (-not (Test-Path $f16Path)) {
    Write-Host "ERROR: F16 model not found at: $f16Path" -ForegroundColor Red
    Write-Host "   Please convert the model to GGUF first." -ForegroundColor Yellow
    exit 1
}

$f16Size = (Get-Item $f16Path).Length / 1GB
Write-Host "F16 Model:" -ForegroundColor Cyan
Write-Host "   Path: $f16Path" -ForegroundColor White
Write-Host "   Size: $([math]::Round($f16Size, 2)) GB" -ForegroundColor White
Write-Host ""

# Output path
$q4Path = "models\medllama-3b-gguf\medllama-3b-q4_k_m.gguf"

Write-Host "Quantizing to Q4_K_M format..." -ForegroundColor Green
Write-Host "   This will take 5-10 minutes..." -ForegroundColor Yellow
Write-Host "   Output: $q4Path" -ForegroundColor White
Write-Host ""

# Run quantization
& $quantizePath $f16Path $q4Path Q4_K_M

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Quantization failed!" -ForegroundColor Red
    exit 1
}

# Check result
if (Test-Path $q4Path) {
    $q4Size = (Get-Item $q4Path).Length / 1GB
    $reduction = (($f16Size - $q4Size) / $f16Size) * 100
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Quantization Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Quantized Model:" -ForegroundColor Cyan
    Write-Host "   Path: $q4Path" -ForegroundColor White
    Write-Host "   Size: $([math]::Round($q4Size, 2)) GB" -ForegroundColor White
    Write-Host "   Reduction: $([math]::Round($reduction, 1))% smaller" -ForegroundColor Green
    Write-Host ""
    Write-Host "Expected Performance:" -ForegroundColor Yellow
    Write-Host "   Generation time: 15-30 seconds (vs 273 seconds)" -ForegroundColor White
    Write-Host "   Tokens/sec: 5-10 (vs 0.9)" -ForegroundColor White
    Write-Host "   Speedup: 5-10x faster!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next Steps:" -ForegroundColor Cyan
    Write-Host "   1. Edit src/config/constants.py line 90:" -ForegroundColor White
    Write-Host "      Change: ./models/medllama-3b-gguf/medllama-3b-f16.gguf" -ForegroundColor Gray
    Write-Host "      To:     ./models/medllama-3b-gguf/medllama-3b-q4_k_m.gguf" -ForegroundColor Gray
    Write-Host "   2. Restart Streamlit app" -ForegroundColor White
    Write-Host "   3. Enjoy 5-10x faster responses!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "ERROR: Quantized file not created. Check errors above." -ForegroundColor Red
}
