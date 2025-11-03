# PowerShell script to set up GGUF conversion for MedLlama
# This script will clone llama.cpp and set up the conversion environment

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🚀 GGUF Setup Script for MedLlama" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right directory
$currentDir = Get-Location
Write-Host "📍 Current directory: $currentDir" -ForegroundColor Yellow

# Step 1: Clone llama.cpp if it doesn't exist
Write-Host "`n📦 Step 1: Checking for llama.cpp..." -ForegroundColor Green

if (Test-Path "llama.cpp") {
    Write-Host "✅ llama.cpp already exists" -ForegroundColor Green
    Set-Location llama.cpp
} else {
    Write-Host "📥 Cloning llama.cpp repository..." -ForegroundColor Yellow
    git clone https://github.com/ggerganov/llama.cpp.git
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to clone llama.cpp. Make sure git is installed." -ForegroundColor Red
        exit 1
    }
    Set-Location llama.cpp
    Write-Host "✅ llama.cpp cloned successfully" -ForegroundColor Green
}

# Step 2: Build llama.cpp
Write-Host "`n🔨 Step 2: Building llama.cpp..." -ForegroundColor Green

if (Test-Path "build\bin\Release\quantize.exe") {
    Write-Host "✅ llama.cpp already built" -ForegroundColor Green
} else {
    Write-Host "📦 Creating build directory..." -ForegroundColor Yellow
    if (-not (Test-Path "build")) {
        New-Item -ItemType Directory -Path "build" | Out-Null
    }
    
    Set-Location build
    Write-Host "🔧 Running CMake..." -ForegroundColor Yellow
    cmake ..
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ CMake failed. Make sure CMake is installed." -ForegroundColor Red
        Write-Host "   Install from: https://cmake.org/download/" -ForegroundColor Yellow
        exit 1
    }
    
    Write-Host "🔨 Building with CMake..." -ForegroundColor Yellow
    cmake --build . --config Release
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Build failed. Check the error messages above." -ForegroundColor Red
        exit 1
    }
    
    Write-Host "✅ llama.cpp built successfully" -ForegroundColor Green
    Set-Location ..
}

# Step 3: Check if Python script exists
Write-Host "`n🐍 Step 3: Checking conversion script..." -ForegroundColor Green

if (Test-Path "convert-hf-to-gguf.py") {
    Write-Host "✅ Conversion script found" -ForegroundColor Green
} else {
    Write-Host "⚠️  convert-hf-to-gguf.py not found in current directory" -ForegroundColor Yellow
    Write-Host "   Make sure you're in the llama.cpp root directory" -ForegroundColor Yellow
    Write-Host "   The script should be at: $(Get-Location)\convert-hf-to-gguf.py" -ForegroundColor Yellow
}

# Step 4: Instructions
Write-Host "`n📋 Step 4: Next Steps" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To convert your model, run:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  # Set your HuggingFace token (if repo is private)" -ForegroundColor White
Write-Host "  `$env:HF_TOKEN='your-token-here'" -ForegroundColor White
Write-Host ""
Write-Host "  # Convert model to GGUF" -ForegroundColor White
Write-Host "  python convert-hf-to-gguf.py AparnaSuresh/MedLlama-3b `\" -ForegroundColor White
Write-Host "    --outdir ..\models\medllama-3b-gguf `\" -ForegroundColor White
Write-Host "    --outfile medllama-3b-f16.gguf" -ForegroundColor White
Write-Host ""
Write-Host "  # Quantize to Q4_K_M (recommended)" -ForegroundColor White
Write-Host "  .\build\bin\Release\quantize.exe `\" -ForegroundColor White
Write-Host "    ..\models\medllama-3b-gguf\medllama-3b-f16.gguf `\" -ForegroundColor White
Write-Host "    ..\models\medllama-3b-gguf\medllama-3b-q4_k_m.gguf `\" -ForegroundColor White
Write-Host "    Q4_K_M" -ForegroundColor White
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan

# Check current location
Write-Host "`n📍 You are now in: $(Get-Location)" -ForegroundColor Cyan
Write-Host "`n✅ Setup complete! You can now convert your model." -ForegroundColor Green

