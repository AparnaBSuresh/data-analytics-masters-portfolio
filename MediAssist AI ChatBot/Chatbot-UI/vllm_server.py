"""
vLLM Server for BioMistral Medical Chatbot
High-performance inference with LoRA adapter support

Requirements:
- NVIDIA GPU with CUDA
- pip install vllm

Run with: python vllm_server.py
"""

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os

# Initialize FastAPI app
app = FastAPI(
    title="BioMistral vLLM API",
    description="High-performance medical LLM inference with vLLM",
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

# Global vLLM engine
llm = None
lora_request = None

# Configuration
CONFIG = {
    "base_model": "BioMistral/BioMistral-7B",
    "lora_adapter_path": "./checkpoint-biomistral",  # Your LoRA adapter
    "hf_token": "",
    "tensor_parallel_size": 1,  # Number of GPUs (1 for single GPU)
    "gpu_memory_utilization": 0.9,  # Use 90% of GPU memory
    "max_model_len": 4096,  # Maximum sequence length
}


class ChatMessage(BaseModel):
    """Chat message model"""
    role: str
    content: str


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    messages: List[ChatMessage]
    department: Optional[str] = "General"
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 256
    top_p: Optional[float] = 0.9


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str
    department: str
    model: str
    tokens_generated: int
    generation_time: float


@app.on_event("startup")
async def load_model():
    """Load vLLM model on startup"""
    global llm, lora_request
    
    print("="*60)
    print("🚀 Loading BioMistral with vLLM...")
    print("="*60)
    print(f"📍 Base model: {CONFIG['base_model']}")
    print(f"🔧 LoRA adapter: {CONFIG['lora_adapter_path']}")
    print(f"💾 GPU memory utilization: {CONFIG['gpu_memory_utilization']*100}%")
    print(f"📏 Max sequence length: {CONFIG['max_model_len']}")
    print()
    
    try:
        # Check if LoRA adapter exists
        lora_path = CONFIG["lora_adapter_path"]
        use_lora = os.path.exists(lora_path) and os.path.exists(os.path.join(lora_path, "adapter_config.json"))
        
        if use_lora:
            print("✅ LoRA adapter found - enabling LoRA support")
            
            # Initialize vLLM with LoRA support
            llm = LLM(
                model=CONFIG["base_model"],
                enable_lora=True,
                max_loras=1,
                max_lora_rank=64,
                tensor_parallel_size=CONFIG["tensor_parallel_size"],
                gpu_memory_utilization=CONFIG["gpu_memory_utilization"],
                max_model_len=CONFIG["max_model_len"],
                trust_remote_code=True,
                download_dir=None,
            )
            
            # Create LoRA request object
            lora_request = LoRARequest(
                lora_name="biomistral_medical",
                lora_int_id=1,
                lora_local_path=lora_path
            )
            
            print("✅ vLLM engine loaded with LoRA adapter!")
            
        else:
            print("⚠️  LoRA adapter not found - loading base model only")
            
            # Initialize vLLM without LoRA
            llm = LLM(
                model=CONFIG["base_model"],
                tensor_parallel_size=CONFIG["tensor_parallel_size"],
                gpu_memory_utilization=CONFIG["gpu_memory_utilization"],
                max_model_len=CONFIG["max_model_len"],
                trust_remote_code=True,
            )
            
            lora_request = None
            print("✅ vLLM engine loaded (base model)!")
        
        print()
        print("="*60)
        print("✅ Server ready for inference!")
        print("="*60)
        print()
        
    except Exception as e:
        print()
        print("="*60)
        print(f"❌ Error loading model: {e}")
        print("="*60)
        print()
        import traceback
        traceback.print_exc()
        raise


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "BioMistral vLLM API",
        "status": "running",
        "model": CONFIG["base_model"],
        "lora_enabled": lora_request is not None,
        "endpoints": {
            "chat": "/api/chat",
            "health": "/health",
            "info": "/info"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": llm is not None,
        "lora_enabled": lora_request is not None
    }


@app.get("/info")
async def model_info():
    """Get model information"""
    return {
        "model": CONFIG["base_model"],
        "lora_adapter": CONFIG["lora_adapter_path"] if lora_request else None,
        "lora_enabled": lora_request is not None,
        "max_model_len": CONFIG["max_model_len"],
        "tensor_parallel_size": CONFIG["tensor_parallel_size"],
        "departments": [
            "Cardiology",
            "Emergency Medicine",
            "Internal Medicine",
            "Pediatrics",
            "Neurology",
            "Oncology",
            "Surgery",
            "Psychiatry",
            "Radiology"
        ]
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for medical questions
    Uses vLLM for high-performance inference
    """
    global llm, lora_request
    
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Extract system and user messages
        system_prompt = ""
        user_message = ""
        
        for msg in request.messages:
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "user":
                user_message = msg.content
        
        # Add department context
        dept_context = f"Department: {request.department}"
        
        # Build prompt
        prompt = f"{system_prompt}\n\n{dept_context}\n\nUser: {user_message}\nAssistant:"
        
        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=["User:", "\n\n\n"],  # Stop sequences
        )
        
        print(f"🔄 Generating response for {request.department} department...")
        print(f"   Temperature: {request.temperature}, Max tokens: {request.max_tokens}")
        
        # Generate with vLLM
        import time
        start_time = time.time()
        
        # Generate with or without LoRA
        if lora_request:
            outputs = llm.generate([prompt], sampling_params, lora_request=lora_request)
        else:
            outputs = llm.generate([prompt], sampling_params)
        
        elapsed = time.time() - start_time
        
        # Extract generated text
        generated_text = outputs[0].outputs[0].text.strip()
        tokens_generated = len(outputs[0].outputs[0].token_ids)
        
        print(f"✅ Generated {tokens_generated} tokens in {elapsed:.2f}s ({tokens_generated/elapsed:.1f} tokens/s)")
        
        return ChatResponse(
            response=generated_text,
            department=request.department,
            model=f"{CONFIG['base_model']}" + (" + LoRA" if lora_request else ""),
            tokens_generated=tokens_generated,
            generation_time=elapsed
        )
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


@app.post("/api/generate")
async def generate(
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9
):
    """
    Simple generation endpoint
    """
    global llm, lora_request
    
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        
        if lora_request:
            outputs = llm.generate([prompt], sampling_params, lora_request=lora_request)
        else:
            outputs = llm.generate([prompt], sampling_params)
        
        generated_text = outputs[0].outputs[0].text.strip()
        
        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "model": CONFIG["base_model"],
            "lora_enabled": lora_request is not None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("Starting vLLM Server...")
    print("="*60)
    print("\n⚠️  IMPORTANT: vLLM requires NVIDIA GPU with CUDA")
    print("   If you don't have a GPU, use the HuggingFace API instead\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

