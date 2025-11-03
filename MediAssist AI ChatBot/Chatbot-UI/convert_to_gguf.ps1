# PowerShell script to convert MedLlama to GGUF format
# Run this from the Chatbot directory

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🔄 MedLlama GGUF Conversion" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if llama.cpp exists
if (-not (Test-Path "llama.cpp")) {
    Write-Host "❌ llama.cpp not found!" -ForegroundColor Red
    Write-Host "   Run: git clone https://github.com/ggerganov/llama.cpp.git" -ForegroundColor Yellow
    exit 1
}

# Set HuggingFace token if needed
$hfToken = $env:HF_TOKEN
if (-not $hfToken) {
    $hfToken = ""
    Write-Host "📝 Using default HF token (set `$env:HF_TOKEN if you want to use a different one)" -ForegroundColor Yellow
}

# Create output directory
$outputDir = "models\medllama-3b-gguf"
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    Write-Host "✅ Created output directory: $outputDir" -ForegroundColor Green
}

Write-Host "`n📦 Step 1: Converting to GGUF format..." -ForegroundColor Green
Write-Host "   This may take 5-10 minutes..." -ForegroundColor Yellow

# Change to llama.cpp directory
Set-Location llama.cpp

# Check if conversion script exists
if (-not (Test-Path "convert-hf-to-gguf.py")) {
    Write-Host "❌ convert-hf-to-gguf.py not found in llama.cpp directory" -ForegroundColor Red
    Write-Host "   Make sure you're using the latest llama.cpp version" -ForegroundColor Yellow
    Set-Location ..
    exit 1
}

# Run conversion
$env:HF_TOKEN = $hfToken
python convert-hf-to-gguf.py AparnaSuresh/MedLlama-3b `
    --outdir "..\$outputDir" `
    --outfile "medllama-3b-f16.gguf"

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n❌ Conversion failed!" -ForegroundColor Red
    Set-Location ..
    exit 1
}

Write-Host "✅ Conversion complete!" -ForegroundColor Green
Set-Location ..

# Check if quantize tool exists
Write-Host "`n🔨 Step 2: Building quantize tool (if needed)..." -ForegroundColor Green

Set-Location llama.cpp

if (-not (Test-Path "build\bin\Release\quantize.exe")) {
    Write-Host "📦 Building quantize tool..." -ForegroundColor Yellow
    
    if (-not (Test-Path "build")) {
        New-Item -ItemType Directory -Path "build" | Out-Null
    }
    
    Set-Location build
    cmake ..
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ CMake failed. Installing CMake might be required." -ForegroundColor Red
        Write-Host "   Download from: https://cmake.org/download/" -ForegroundColor Yellow
        Set-Location ..\..
        exit 1
    }
    
    cmake --build . --config Release
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Build failed. Check error messages above." -ForegroundColor Red
        Set-Location ..\..
        exit 1
    }
    
    Set-Location ..
}

Set-Location ..

Write-Host "✅ Quantize tool ready" -ForegroundColor Green

# Quantize the model
Write-Host "`n⚡ Step 3: Quantizing to Q4_K_M (recommended)..." -ForegroundColor Green
Write-Host "   This may take a few minutes..." -ForegroundColor Yellow

$f16Path = "$outputDir\medllama-3b-f16.gguf"
$q4Path = "$outputDir\medllama-3b-q4_k_m.gguf"

if (-not (Test-Path $f16Path)) {
    Write-Host "❌ F16 model not found at: $f16Path" -ForegroundColor Red
    exit 1
}

Set-Location llama.cpp
.\build\bin\Release\quantize.exe "..\$f16Path" "..\$q4Path" Q4_K_M

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Quantization failed!" -ForegroundColor Red
    Set-Location ..
    exit 1
}

Set-Location ..

# Verify output
if (Test-Path $q4Path) {
    $fileSize = (Get-Item $q4Path).Length / 1MB
    Write-Host "`n✅ Quantization complete!" -ForegroundColor Green
    Write-Host "   Output file: $q4Path" -ForegroundColor Cyan
    Write-Host "   File size: $([math]::Round($fileSize, 2)) MB" -ForegroundColor Cyan
    
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "🎉 Conversion Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Your GGUF model is ready at:" -ForegroundColor Yellow
    Write-Host "  $q4Path" -ForegroundColor White
    Write-Host ""
    Write-Host "The app will automatically use it!" -ForegroundColor Green
    Write-Host "Just run: streamlit run app.py" -ForegroundColor Cyan
} else {
    Write-Host "❌ Quantized model not found. Check for errors above." -ForegroundColor Red
}

