# Quick script to check build status

$env:PATH += ";C:\Program Files\CMake\bin"

Write-Host "Checking build status..." -ForegroundColor Cyan
Write-Host ""

# Check for quantize tool in common locations
$locations = @(
    "llama.cpp\build\bin\Release\quantize.exe",
    "llama.cpp\build\tools\quantize\Release\quantize.exe",
    "llama.cpp\build\Release\quantize.exe",
    "llama.cpp\build\tools\quantize\quantize.exe"
)

$found = $false
foreach ($loc in $locations) {
    if (Test-Path $loc) {
        $file = Get-Item $loc
        Write-Host "✅ BUILD COMPLETE!" -ForegroundColor Green
        Write-Host "   Location: $($file.FullName)" -ForegroundColor Cyan
        Write-Host "   Size: $([math]::Round($file.Length / 1MB, 2)) MB" -ForegroundColor Yellow
        Write-Host "   Modified: $($file.LastWriteTime)" -ForegroundColor Yellow
        $found = $true
        break
    }
}

if (-not $found) {
    Write-Host "⏳ Build still running or not started" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Check if msbuild is running:" -ForegroundColor Cyan
    $msbuild = Get-Process msbuild -ErrorAction SilentlyContinue
    if ($msbuild) {
        Write-Host "   ✅ msbuild is running (build in progress)" -ForegroundColor Green
    } else {
        Write-Host "   ❌ msbuild not running" -ForegroundColor Red
        Write-Host ""
        Write-Host "To start build, run:" -ForegroundColor Yellow
        Write-Host "   cd llama.cpp\build" -ForegroundColor White
        Write-Host "   `$env:PATH += ';C:\Program Files\CMake\bin'" -ForegroundColor White
        Write-Host "   cmake --build . --config Release" -ForegroundColor White
    }
}

