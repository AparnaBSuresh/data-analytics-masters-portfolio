"""
FastAPI server for MedLlama chatbot using HuggingFace Inference API.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from src.llm.backend_hf import call_hf_model

app = FastAPI(
    title="MedLlama Chat API",
    description="Healthcare chatbot using MedLlama-3b via HuggingFace Inference API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    """Chat message model"""
    role: str
    content: str


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    model_key: str = "MedLlama-HF-API"
    messages: List[ChatMessage]
    max_new_tokens: int = 512
    temperature: float = 0.2


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    content: str
    model: str


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "MedLlama Chat API",
        "status": "running",
        "model": "AparnaSuresh/MedLlama-3b",
        "endpoints": {
            "chat": "/api/chat",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "backend": "HuggingFace Inference API"
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Chat endpoint for medical questions.
    """
    try:
        # Convert Pydantic models to dicts for backend
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        
        # Call HF model
        text = call_hf_model(
            req.model_key,
            messages,
            req.max_new_tokens,
            req.temperature
        )
        
        return ChatResponse(
            content=text,
            model="MedLlama-3b"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 Starting MedLlama API Server")
    print("="*60)
    print("\n⚠️  Make sure HF_TOKEN is set in your environment:")
    print("   PowerShell: $env:HF_TOKEN='your-token'")
    print("   Linux/Mac:  export HF_TOKEN='your-token'\n")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=7000)


