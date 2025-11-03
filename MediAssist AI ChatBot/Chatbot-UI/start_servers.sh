#!/bin/bash
# Startup script for BioMistral Chatbot with FastAPI backend
# This script starts both the FastAPI server and Streamlit app

echo "========================================"
echo "BioMistral Medical Chatbot Startup"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 is not installed"
    exit 1
fi

echo "[1/3] Starting FastAPI server..."
echo "This will load your BioMistral model (takes 2-5 minutes)"
echo ""

# Start FastAPI in background
nohup uvicorn fastapi_server:app --host 0.0.0.0 --port 8000 > fastapi.log 2>&1 &
FASTAPI_PID=$!
echo "FastAPI server started (PID: $FASTAPI_PID)"
echo "Logs: tail -f fastapi.log"

echo ""
echo "Waiting for FastAPI server to start..."
sleep 10

echo ""
echo "[2/3] Checking FastAPI server status..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "SUCCESS: FastAPI server is running!"
else
    echo "WARNING: FastAPI server not responding yet"
    echo "It may still be loading the model..."
    echo "Check logs: tail -f fastapi.log"
fi

echo ""
echo "[3/3] Starting Streamlit app..."
sleep 2

# Start Streamlit in background
nohup streamlit run app.py > streamlit.log 2>&1 &
STREAMLIT_PID=$!
echo "Streamlit app started (PID: $STREAMLIT_PID)"
echo "Logs: tail -f streamlit.log"

echo ""
echo "========================================"
echo "Servers Started!"
echo "========================================"
echo ""
echo "FastAPI Server: http://localhost:8000"
echo "Streamlit App:  http://localhost:8501"
echo ""
echo "IMPORTANT:"
echo "1. Wait for 'Model loaded successfully' in fastapi.log"
echo "2. Then open http://localhost:8501 in your browser"
echo "3. Select 'BioMistral-7B' and start chatting!"
echo ""
echo "Process IDs:"
echo "  FastAPI:   $FASTAPI_PID"
echo "  Streamlit: $STREAMLIT_PID"
echo ""
echo "To stop servers:"
echo "  kill $FASTAPI_PID $STREAMLIT_PID"
echo ""
echo "To view logs:"
echo "  tail -f fastapi.log"
echo "  tail -f streamlit.log"
echo "========================================"
echo ""


