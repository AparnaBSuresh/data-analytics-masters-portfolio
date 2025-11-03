@echo off
REM Startup script for BioMistral Chatbot with FastAPI backend
REM This script starts both the FastAPI server and Streamlit app

echo ========================================
echo BioMistral Medical Chatbot Startup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo [1/3] Starting FastAPI server...
echo This will load your BioMistral model (takes 2-5 minutes)
echo.

REM Start FastAPI in a new window
start "FastAPI Server" cmd /k "uvicorn fastapi_server:app --host 0.0.0.0 --port 8000"

echo Waiting for FastAPI server to start...
timeout /t 10 /nobreak >nul

echo.
echo [2/3] Checking FastAPI server status...
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo WARNING: FastAPI server not responding yet
    echo It may still be loading the model...
    echo Check the FastAPI Server window for progress
) else (
    echo SUCCESS: FastAPI server is running!
)

echo.
echo [3/3] Starting Streamlit app...
timeout /t 2 /nobreak >nul

REM Start Streamlit in a new window
start "Streamlit App" cmd /k "streamlit run app.py"

echo.
echo ========================================
echo Servers Started!
echo ========================================
echo.
echo FastAPI Server: http://localhost:8000
echo Streamlit App:  http://localhost:8501
echo.
echo IMPORTANT:
echo 1. Wait for "Model loaded successfully" in FastAPI window
echo 2. Then open http://localhost:8501 in your browser
echo 3. Select "BioMistral-7B" and start chatting!
echo.
echo To stop: Close both command windows
echo ========================================
echo.
pause


