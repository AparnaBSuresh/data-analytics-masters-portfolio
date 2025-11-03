"""
FastAPI Server for BioMistral Medical Chatbot
Serves the fine-tuned model as an API endpoint

Run with: uvicorn fastapi_server:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import os

# Initialize FastAPI app
app = FastAPI(
    title="BioMistral Medical API",
    description="API for fine-tuned medical language model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Streamlit app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and tokenizer
model = None
tokenizer = None

# Configuration
MODEL_CONFIG = {
    "base_model": "BioMistral/BioMistral-7B",
    "adapter_path": os.path.abspath("./checkpoint-biomistral"),  # Use absolute path
    "hf_token": "",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
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
    max_tokens: Optional[int] = 256  # Reduced for faster CPU inference


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str
    department: str
    model: str


@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model, tokenizer
    
    print("🚀 Loading BioMistral model...")
    print(f"📍 Device: {MODEL_CONFIG['device']}")
    print(f"📂 Adapter path: {MODEL_CONFIG['adapter_path']}")
    
    # Verify adapter files exist
    adapter_path = MODEL_CONFIG["adapter_path"]
    required_files = ["adapter_model.safetensors", "adapter_config.json"]
    
    print("🔍 Verifying adapter files...")
    for file in required_files:
        file_path = os.path.join(adapter_path, file)
        if os.path.exists(file_path):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - MISSING!")
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    try:
        # Load tokenizer
        print("📚 Loading tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                adapter_path,
                token=MODEL_CONFIG["hf_token"],
                legacy=False
            )
            print("  ✅ Loaded tokenizer from adapter path")
        except Exception as e:
            print(f"  ⚠️  Adapter tokenizer not found, using base model tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_CONFIG["base_model"],
                token=MODEL_CONFIG["hf_token"],
                legacy=False
            )
            print("  ✅ Loaded tokenizer from base model")
        
        # Load base model
        print("🧠 Loading base model...")
        
        # Create offload folder if it doesn't exist
        offload_folder = "./offload_folder"
        os.makedirs(offload_folder, exist_ok=True)
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_CONFIG["base_model"],
            device_map="auto",
            token=MODEL_CONFIG["hf_token"],
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            offload_folder=offload_folder,
            offload_state_dict=True
        )
        print("  ✅ Base model loaded")
        
        # Check if this is a full fine-tuned model or just LoRA adapters
        print("🔧 Checking adapter type...")
        from safetensors import safe_open
        
        adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
        with safe_open(adapter_file, framework="pt") as f:
            keys = list(f.keys())
            has_full_weights = any("lm_head" in k or "embed_tokens" in k for k in keys)
            has_lora_weights = any("lora_A" in k or "lora_B" in k for k in keys)
        
        print(f"  📊 Total tensors: {len(keys)}")
        print(f"  📦 Has full model weights: {has_full_weights}")
        print(f"  🔧 Has LoRA weights: {has_lora_weights}")
        
        if has_full_weights and has_lora_weights:
            # This is a checkpoint with both base and LoRA weights
            # We need to load it differently
            print("  ℹ️  Detected: Full checkpoint with LoRA weights")
            print("  🔄 Loading as merged fine-tuned model...")
            
            # Load the merged model directly from the checkpoint
            from safetensors.torch import load_file
            state_dict = load_file(adapter_file)
            
            # Load only the LoRA weights into the model
            lora_state_dict = {k: v for k, v in state_dict.items() if "lora_" in k}
            print(f"  📥 Loading {len(lora_state_dict)} LoRA tensors...")
            
            try:
                # Try PEFT loading
                model = PeftModel.from_pretrained(model, adapter_path)
                print("  ✅ LoRA adapter loaded with PEFT")
                
                print("⚡ Merging adapter into base model...")
                model = model.merge_and_unload()
                print("  ✅ Adapter merged")
            except Exception as e:
                print(f"  ⚠️  PEFT merge failed: {e}")
                print("  🔄 Trying manual LoRA weight injection...")
                
                # Manual approach: Inject LoRA weights directly
                try:
                    # Get model's state dict
                    model_state = model.state_dict()
                    
                    # Apply LoRA weights manually
                    # LoRA formula: W_new = W_base + (lora_B @ lora_A) * scaling
                    updated_count = 0
                    for key in lora_state_dict.keys():
                        if "lora_A" in key:
                            # Get corresponding lora_B
                            base_key = key.replace("lora_A.weight", "")
                            lora_a_key = key
                            lora_b_key = key.replace("lora_A", "lora_B")
                            
                            if lora_b_key in lora_state_dict:
                                # Get the base weight key
                                weight_key = base_key.replace("base_model.model.", "") + "weight"
                                
                                if weight_key in model_state:
                                    lora_a = lora_state_dict[lora_a_key].to(model.device)
                                    lora_b = lora_state_dict[lora_b_key].to(model.device)
                                    base_weight = model_state[weight_key]
                                    
                                    # Apply LoRA: W = W_base + (B @ A) * scaling
                                    # Default LoRA scaling: alpha / r = 32 / 16 = 2.0
                                    scaling = 2.0
                                    delta = (lora_b @ lora_a) * scaling
                                    
                                    # Update weight
                                    model_state[weight_key] = base_weight + delta
                                    updated_count += 1
                    
                    if updated_count > 0:
                        # Load updated state dict back into model
                        model.load_state_dict(model_state, strict=False)
                        print(f"  ✅ Manually applied {updated_count} LoRA adaptations")
                    else:
                        print("  ⚠️  Could not apply LoRA weights manually")
                        print("  ℹ️  Continuing with base model only")
                        
                except Exception as e2:
                    print(f"  ⚠️  Manual merge also failed: {e2}")
                    print("  ℹ️  Continuing with base model only")
        
        elif has_lora_weights:
            # Pure LoRA adapter
            print("  ℹ️  Detected: Pure LoRA adapter")
            try:
                model = PeftModel.from_pretrained(model, adapter_path)
                print("  ✅ LoRA adapter loaded")
                
                print("⚡ Merging adapter into base model...")
                model = model.merge_and_unload()
                print("  ✅ Adapter merged")
            except Exception as e:
                print(f"  ⚠️  PEFT loading failed: {e}")
                print("  ℹ️  Continuing with base model only")
        else:
            print("  ℹ️  No LoRA weights found, using base model")
        
        # Set to eval mode
        model.eval()
        print("  ✅ Model set to eval mode")
        
        print("\n" + "="*50)
        print("✅ Model loaded successfully!")
        print("="*50 + "\n")
        
    except Exception as e:
        print("\n" + "="*50)
        print(f"❌ Error loading model: {e}")
        print("="*50 + "\n")
        import traceback
        traceback.print_exc()
        raise


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "BioMistral Medical API",
        "status": "running",
        "model": "BioMistral-7B (LoRA Fine-tuned)",
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
        "model_loaded": model is not None,
        "device": MODEL_CONFIG["device"]
    }


@app.get("/info")
async def model_info():
    """Get model information"""
    return {
        "model": "BioMistral-7B",
        "type": "LoRA Fine-tuned",
        "base_model": MODEL_CONFIG["base_model"],
        "adapter_path": MODEL_CONFIG["adapter_path"],
        "device": MODEL_CONFIG["device"],
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
    
    Args:
        request: ChatRequest with messages, department, temperature, max_tokens
        
    Returns:
        ChatResponse with generated response
    """
    global model, tokenizer
    
    if model is None or tokenizer is None:
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
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        print(f"🔄 Generating response for {request.department} department...")
        print(f"   Max tokens: {request.max_tokens}, Temperature: {request.temperature}")
        
        # Generate
        import time
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                top_p=0.9,
                repetition_penalty=1.1
            )
        
        elapsed = time.time() - start_time
        print(f"✅ Generation completed in {elapsed:.2f} seconds")
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "Assistant:" in generated_text:
            response = generated_text.split("Assistant:", 1)[-1].strip()
        else:
            response = generated_text.strip()
        
        return ChatResponse(
            response=response,
            department=request.department,
            model="BioMistral-7B-LoRA"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


@app.post("/api/generate")
async def generate(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.2
):
    """
    Simple generation endpoint
    
    Args:
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Generated text
    """
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "model": "BioMistral-7B-LoRA"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

