"""
LLM backend implementations for different providers.
"""

import os
import json
import time
from typing import List, Dict, Tuple, Optional


# Global variable to store last timing info (for UI access)
_last_timing_info: Optional[Dict] = None

def get_last_timing_info() -> Optional[Dict]:
    """Get the timing info from the last LLM call."""
    return _last_timing_info

def call_llm(messages: List[Dict],
             provider: str,
             model_name: str,
             temperature: float = 0.2,
             api_key: str = "",
             api_base: str = "",
             hf_model_path: str = "",
             hf_token: str = "",
             max_new_tokens: int = 512,
             department: str = "General") -> str:
    """
    Call LLM with different backend providers including fine-tuned medical models.
    
    Backends:
      - provider == 'OpenAI'  : OpenAI Chat Completions (fine‑tuned or base)
      - provider == 'Together': Together API
      - provider == 'HuggingFace (local)': local Transformers pipeline with fine‑tuned models
      - provider == 'Fine-tuned Models': BioMistral-7B, MedLlama-3, Meditron, MedAlpaca
      - provider == 'Debug/Echo': Debug mode that echoes back the user input
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        provider: Backend provider name
        model_name: Model name or repository ID
        temperature: Sampling temperature
        api_key: API key for the provider
        api_base: Custom API base URL
        hf_model_path: Local path for HuggingFace models
        max_new_tokens: Maximum tokens to generate
        department: Medical department specialization
        
    Returns:
        Generated response text
    """
    try:
        if provider == "OpenAI":
            return _call_openai(messages, model_name, temperature, api_key, api_base)
        elif provider == "Together":
            return _call_together(messages, model_name, temperature, api_key, api_base)
        elif provider == "HuggingFace (local)":
            return _call_huggingface_local(messages, model_name, temperature, hf_model_path, max_new_tokens, hf_token)
        elif provider == "Fine-tuned Models":
            return _call_finetuned_model(messages, model_name, temperature, department, max_new_tokens, hf_token)
        else:
            return _call_debug_echo(messages)

    except Exception as e:
        return f"(Model error) {e}\nTip: Check your keys/paths in the sidebar."


def _call_openai(messages: List[Dict], model_name: str, temperature: float, api_key: str, api_base: str) -> str:
    """Call OpenAI Chat Completions API."""
    from openai import OpenAI
    
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key: 
        raise RuntimeError("Missing OPENAI_API_KEY")
    
    client = OpenAI(api_key=key, base_url=(api_base or None))
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content


def _call_together(messages: List[Dict], model_name: str, temperature: float, api_key: str, api_base: str) -> str:
    """Call Together API."""
    import requests
    
    key = api_key or os.environ.get("TOGETHER_API_KEY")
    if not key: 
        raise RuntimeError("Missing TOGETHER_API_KEY")
    
    url = api_base or "https://api.together.xyz/v1/chat/completions"
    payload = {
        "model": model_name, 
        "messages": messages, 
        "temperature": temperature, 
        "stream": False
    }
    headers = {
        "Authorization": f"Bearer {key}", 
        "Content-Type": "application/json"
    }
    
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


def _call_huggingface_local(messages: List[Dict], model_name: str, temperature: float, hf_model_path: str, max_new_tokens: int, hf_token: str = "") -> str:
    """Call local HuggingFace Transformers pipeline."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    
    model_id = hf_model_path or model_name
    
    # Get token from parameter or environment variable
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    
    # Create offload folder if needed
    offload_folder = "./offload_folder"
    os.makedirs(offload_folder, exist_ok=True)
    
    tok = AutoTokenizer.from_pretrained(model_id, token=token)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", 
        token=token,
        offload_folder=offload_folder,
        low_cpu_mem_usage=True
    )
    pipe = pipeline("text-generation", model=mdl, tokenizer=tok)
    
    # Convert chat messages into a single prompt (simple template)
    sys = ""
    user = ""
    for m in messages:
        if m["role"] == "system": 
            sys = m["content"]
        if m["role"] == "user":   
            user = m["content"]
    
    prompt = (sys + "\n\nUser: " + user + "\nAssistant:").strip()
    out = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
    return out[0]["generated_text"].split("Assistant:", 1)[-1].strip()


def _call_finetuned_model(messages: List[Dict], model_name: str, temperature: float, department: str, max_new_tokens: int, hf_token: str = "") -> str:
    """Call fine-tuned medical models with department-specific context."""
    from ..config.constants import FINETUNED_MODELS, MEDICAL_DEPARTMENTS
    
    # Get model configuration
    if model_name not in FINETUNED_MODELS:
        raise ValueError(f"Unknown fine-tuned model: {model_name}")
    
    model_config = FINETUNED_MODELS[model_name]
    use_api = model_config.get("use_api", False)
    use_gguf = model_config.get("use_gguf", False)
    
    # Get department context
    dept_info = MEDICAL_DEPARTMENTS.get(department, {})
    dept_context = f"Department: {department}"
    if dept_info:
        dept_context += f" - {dept_info.get('specialization', '')}"
    
    # Get token from parameter or environment variable
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    
    # Choose backend: GGUF (fastest) > API > Local transformers
    if use_gguf:
        # Use GGUF backend with llama.cpp (fastest on CPU)
        try:
            from .backend_gguf import call_gguf_model
            model_path = model_config.get("model_path")
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"GGUF model not found at: {model_path}\n\n"
                    "Please convert your model to GGUF format first:\n"
                    "1. See quantize_to_gguf.py for instructions\n"
                    "2. Or download a pre-converted GGUF model\n"
                    "3. Update model_path in constants.py"
                )
            
            # Add department context to system message
            enhanced_messages = messages.copy()
            if enhanced_messages and enhanced_messages[0].get("role") == "system":
                enhanced_messages[0]["content"] = f"{enhanced_messages[0]['content']}\n\n{dept_context}"
            else:
                enhanced_messages.insert(0, {"role": "system", "content": dept_context})
            
            response, timing_info = call_gguf_model(
                enhanced_messages,
                model_path=model_path,
                max_tokens=max_new_tokens,
                temperature=temperature
            )
            
            # Store timing info globally so UI can access it
            global _last_timing_info
            _last_timing_info = timing_info
            return response
            
        except ImportError as e:
            error_msg = f"❌ **llama-cpp-python not installed.**\n\nInstall it with:\n```bash\npip install llama-cpp-python\n```\n\nError: {str(e)}"
            print(f"❌ Import Error: {e}")
            return error_msg
        except Exception as e:
            error_msg = f"❌ **GGUF backend error:** {str(e)}\n\n**Debug Info:**\n- Model path: {model_path}\n- Path exists: {os.path.exists(model_path) if model_path else 'N/A'}\n- Use GGUF: {use_gguf}"
            print(f"❌ GGUF Backend Error: {e}")
            import traceback
            print(traceback.format_exc())
            return error_msg
    
    elif use_api:
        try:
            return _call_huggingface_api(messages, model_config, dept_context, temperature, max_new_tokens, token, department, dept_info)
        except Exception as api_error:
            # If API fails, try local loading as fallback
            print(f"⚠️ API call failed: {api_error}")
            print("🔄 Falling back to local model loading...")
            if model_config.get("model_path"):
                return _call_local_model(messages, model_config, dept_context, temperature, max_new_tokens, token, department, dept_info)
            else:
                raise RuntimeError(f"API failed and no local model path configured: {api_error}")
    else:
        return _call_local_model(messages, model_config, dept_context, temperature, max_new_tokens, token, department, dept_info)


def _call_huggingface_api(messages: List[Dict], model_config: Dict, dept_context: str, temperature: float, max_new_tokens: int, token: str, department: str, dept_info: Dict) -> str:
    """Call FastAPI or HuggingFace Inference API."""
    import requests
    
    api_url = model_config.get("api_url")
    if not api_url:
        raise ValueError("API URL not configured for this model")
    
    # Check API type
    is_fastapi = "localhost" in api_url or "127.0.0.1" in api_url or "/api/chat" in api_url
    is_hf_inference = model_config.get("use_hf_inference", False) or "huggingface.co" in api_url
    
    if is_hf_inference:
        # Use the new backend_hf module with proper chat template
        try:
            from .backend_hf import call_hf_model
            
            # Add department context to system message if not present
            enhanced_messages = messages.copy()
            if enhanced_messages and enhanced_messages[0].get("role") == "system":
                enhanced_messages[0]["content"] = f"{enhanced_messages[0]['content']}\n\n{dept_context}"
            else:
                enhanced_messages.insert(0, {"role": "system", "content": dept_context})
            
            # Get model key by matching api_url
            from ..config.constants import FINETUNED_MODELS
            model_key = None
            for key, cfg in FINETUNED_MODELS.items():
                if cfg.get("api_url") == api_url:
                    model_key = key
                    break
            
            if not model_key:
                model_key = "MedLlama-HF-API"  # Default fallback
            
            generated_text = call_hf_model(
                model_key,
                enhanced_messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            
            return generated_text
            
        except ImportError:
            # Fallback to old method if backend_hf not available
            pass
        except Exception as e:
            # If new backend fails, try old method
            pass
        
        # Fallback: Original method (without chat template)
        if not token:
            return "⚠️ **HuggingFace token required**\n\nPlease set your HuggingFace token to use the cloud API."
        
        # Build prompt from messages
        sys = ""
        user = ""
        for m in messages:
            if m["role"] == "system": 
                sys = m["content"]
            if m["role"] == "user":   
                user = m["content"]
        
        # Enhanced prompt with department context
        prompt = f"{sys}\n\n{dept_context}\n\nUser: {user}\nAssistant:"
        
        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": min(max_new_tokens, 250),
                "temperature": temperature,
                "do_sample": True,
                "return_full_text": False,
                "top_p": 0.9
            }
        }
        
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            result = response.json()
            
            # Extract generated text
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
            elif isinstance(result, dict):
                generated_text = result.get("generated_text", "") or result.get("generated", "")
            else:
                generated_text = str(result)
            
            return generated_text
            
        except requests.exceptions.RequestException as e:
            return f"⚠️ **HuggingFace API Error:** {str(e)}\n\nThe model may be loading (first request takes 20-30 seconds). Please try again."
    
    elif is_fastapi:
        # Call FastAPI endpoint
        payload = {
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "department": department,
            "temperature": temperature,
            "max_tokens": max_new_tokens
        }
        
        try:
            # Increased timeout for CPU inference (can take 30-60 seconds per request)
            response = requests.post(api_url, json=payload, timeout=300)  # 5 minutes
            response.raise_for_status()
            result = response.json()
            
            generated_text = result.get("response", "")
            
            return generated_text
            
        except requests.exceptions.RequestException as e:
            return f"API Error: {str(e)}\n\nTip: Make sure FastAPI server is running (uvicorn fastapi_server:app --port 8000)"
    
    else:
        # Call HuggingFace Inference API
        if not token:
            raise RuntimeError("HuggingFace token required for API access")
        
        # Build prompt from messages
        sys = ""
        user = ""
        for m in messages:
            if m["role"] == "system": 
                sys = m["content"]
            if m["role"] == "user":   
                user = m["content"]
        
        # Enhanced prompt with department context
        prompt = f"{sys}\n\n{dept_context}\n\nUser: {user}\nAssistant:"
        
        # Call HuggingFace Inference API
        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            # Extract generated text
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
            else:
                generated_text = result.get("generated_text", "")
            
            return generated_text
            
        except requests.exceptions.RequestException as e:
            return f"API Error: {str(e)}\n\nTip: Check your HuggingFace token and API endpoint."


# Global cache for loaded models (keyed by model_path)
_model_cache = {}
_tokenizer_cache = {}

def _call_local_model(messages: List[Dict], model_config: Dict, dept_context: str, temperature: float, max_new_tokens: int, token: str, department: str, dept_info: Dict) -> str:
    """Load and run model locally from HuggingFace or local path."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    model_path = model_config["model_path"]
    base_model = model_config.get("base_model")
    is_adapter = model_config.get("is_adapter", False)
    
    # Create offload folder if needed
    offload_folder = "./offload_folder"
    os.makedirs(offload_folder, exist_ok=True)
    
    # Check cache first
    cache_key = f"{model_path}_{is_adapter}"
    if cache_key in _model_cache:
        mdl = _model_cache[cache_key]
        tok = _tokenizer_cache[cache_key]
    else:
        print(f"🔄 Loading model from: {model_path}")
        print("⏳ This may take a few minutes on first load...")
        
        # Load model - check if it's a LoRA adapter or full model
        if is_adapter and base_model:
            # Load base model + LoRA adapter
            try:
                from peft import PeftModel
                
                # Load tokenizer from adapter path or base model
                try:
                    tok = AutoTokenizer.from_pretrained(model_path, token=token, use_fast=True)
                except:
                    tok = AutoTokenizer.from_pretrained(base_model, token=token, use_fast=True)
                
                # Load base model
                print(f"📦 Loading base model: {base_model}")
                mdl = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    device_map="auto",
                    token=token,
                    offload_folder=offload_folder,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                
                # Load and merge LoRA adapter
                print(f"🔧 Loading LoRA adapter: {model_path}")
                mdl = PeftModel.from_pretrained(mdl, model_path, token=token)
                mdl = mdl.merge_and_unload()  # Merge adapter weights into base model
                
            except ImportError:
                raise RuntimeError("LoRA adapter detected but 'peft' library not installed. Run: pip install peft")
        else:
            # Load full fine-tuned model (can be HuggingFace repo ID or local path)
            print(f"📦 Loading model: {model_path}")
            tok = AutoTokenizer.from_pretrained(model_path, token=token, use_fast=True)
            mdl = AutoModelForCausalLM.from_pretrained(
                model_path, 
                device_map="auto", 
                token=token,
                offload_folder=offload_folder,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        
        mdl.eval()  # Set to evaluation mode
        
        # Cache the loaded model
        _model_cache[cache_key] = mdl
        _tokenizer_cache[cache_key] = tok
        print("✅ Model loaded and cached!")
    
    # Add department context to system message
    enhanced_messages = messages.copy()
    if enhanced_messages and enhanced_messages[0].get("role") == "system":
        enhanced_messages[0]["content"] = f"{enhanced_messages[0]['content']}\n\n{dept_context}"
    else:
        enhanced_messages.insert(0, {"role": "system", "content": dept_context})
    
    # Format prompt using chat template if available, otherwise use simple format
    prompt = None
    if hasattr(tok, "chat_template") and tok.chat_template is not None:
        try:
            prompt = tok.apply_chat_template(
                enhanced_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            print(f"⚠️ Chat template failed: {e}, using fallback format")
    
    # Fallback to Llama-style format (works well for most models)
    if prompt is None:
        sys = ""
        user = ""
        assistant = ""
        for m in enhanced_messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system": 
                sys = content
            elif role == "user":   
                user = content
            elif role == "assistant":
                assistant = content
        
        # Try simpler format first (many models work better with this)
        if sys:
            # Simple format: System message, then user question
            prompt = f"System: {sys}\n\nUser: {user}\n\nAssistant:"
        else:
            prompt = f"User: {user}\n\nAssistant:"
        
        # Store alternative Llama format in case we need to retry
        llama_prompt = None
        if sys:
            llama_prompt = f"[INST] <<SYS>>\n{sys}\n<</SYS>>\n\n{user} [/INST]"
        else:
            llama_prompt = f"[INST] {user} [/INST]"
        
        # If assistant content exists (for conversation history), add it
        if assistant:
            prompt = f"{prompt}\n{assistant}\n\nUser: {user}\n\nAssistant:"
            llama_prompt = f"{llama_prompt}\n{assistant}\n[INST] {user} [/INST]"
    
    print(f"📝 Using prompt format (first 200 chars): {prompt[:200]}...")
    
    # Set pad token if not set
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    
    # Tokenize and generate
    print(f"🔤 Tokenizing prompt... (length: {len(prompt)} chars)")
    inputs = tok(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    
    # Move inputs to model device
    # Handle device_map="auto" where model might be on multiple devices
    try:
        # Try to get device from first parameter
        model_device = next(mdl.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        print(f"📍 Moving inputs to device: {model_device}")
    except Exception as e:
        # Fallback: keep on CPU or try cuda if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            print("📍 Moving inputs to CUDA")
        else:
            print("📍 Keeping inputs on CPU")
    
    # Generate response
    # For very low temperature, use greedy decoding; otherwise use sampling
    use_sample = temperature > 0.1
    min_new_tokens = max(20, max_new_tokens // 10)  # Ensure at least some tokens
    
    print(f"🧠 Generating response (max_new_tokens={max_new_tokens}, temperature={temperature}, sampling={use_sample})...")
    try:
        with torch.no_grad():
            # Build generation kwargs
            gen_kwargs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs.get("attention_mask"),
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tok.pad_token_id,
                "eos_token_id": tok.eos_token_id,
                "repetition_penalty": 1.1,
            }
            
            # Try to add min_new_tokens if supported (test by checking generate signature)
            import inspect
            sig = inspect.signature(mdl.generate)
            if "min_new_tokens" in sig.parameters:
                gen_kwargs["min_new_tokens"] = min_new_tokens
            
            # Add sampling parameters only if using sampling
            if use_sample:
                gen_kwargs.update({
                    "temperature": max(temperature, 0.3),  # Don't let temperature be too low
                    "do_sample": True,
                    "top_p": 0.9,
                    "top_k": 50,
                })
            else:
                # Greedy decoding for low temperature
                gen_kwargs["do_sample"] = False
            
            outputs = mdl.generate(**gen_kwargs)
        
        new_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
        print(f"✅ Generated {new_tokens} new tokens")
        
        # If we got very few tokens, try again with adjusted parameters
        if new_tokens < 5:
            print(f"⚠️ Got only {new_tokens} tokens, retrying with greedy decoding and higher min_new_tokens...")
            gen_kwargs["do_sample"] = False
            gen_kwargs.pop("temperature", None)
            gen_kwargs.pop("top_p", None)
            gen_kwargs.pop("top_k", None)
            if "min_new_tokens" in gen_kwargs:
                gen_kwargs["min_new_tokens"] = 30  # Force at least 30 tokens
            # Also try increasing max_new_tokens slightly
            gen_kwargs["max_new_tokens"] = max(max_new_tokens, 100)
            outputs = mdl.generate(**gen_kwargs)
            new_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
            print(f"✅ Retry generated {new_tokens} new tokens")
            
            # If still getting very few tokens, the prompt format might be wrong
            if new_tokens < 10:
                print("⚠️ Still getting few tokens - prompt format may need adjustment")
            
    except Exception as e:
        error_msg = f"❌ Generation error: {str(e)}"
        print(error_msg)
        return f"**Error generating response:** {str(e)}\n\nPlease check:\n- Model is properly loaded\n- Sufficient memory available\n- Tokenizer matches model"
    
    # Decode the generated tokens
    input_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_length:]
    
    # Try decoding with special tokens first (to see what we get)
    response = tok.decode(generated_ids, skip_special_tokens=False)
    
    # Remove special tokens manually if needed
    if hasattr(tok, 'eos_token') and tok.eos_token:
        response = response.replace(tok.eos_token, "")
    if hasattr(tok, 'pad_token') and tok.pad_token:
        response = response.replace(tok.pad_token, "")
    response = response.strip()
    
    # Also try with skip_special_tokens=True
    response_clean = tok.decode(generated_ids, skip_special_tokens=True).strip()
    
    # Use whichever gives us more content
    if len(response_clean) > len(response):
        response = response_clean
    
    # If response is still empty or very short, try decoding full output
    if not response or len(response) < 10:
        print("⚠️ New tokens empty, trying full output decode...")
        full_response = tok.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant part based on prompt format used
        if "[INST]" in prompt and "[/INST]" in prompt:
            # Llama format - extract after [/INST]
            parts = full_response.split("[/INST]")
            if len(parts) > 1:
                response = parts[-1].strip()
        elif "### Assistant:" in full_response:
            response = full_response.split("### Assistant:")[-1].strip()
        elif "Assistant:" in full_response:
            response = full_response.split("Assistant:")[-1].strip()
        else:
            # Remove the input prompt from the full response
            if prompt in full_response:
                response = full_response.split(prompt)[-1].strip()
            else:
                # Just return the last part of the response
                response = full_response[-500:].strip() if len(full_response) > 500 else full_response.strip()
    
    # Final check - if still empty, provide debug info
    if not response or len(response.strip()) < 5:
        # Get model device safely
        try:
            model_device = str(next(mdl.parameters()).device)
        except:
            model_device = "unknown"
        
        # Get full decoded output for debugging
        try:
            full_output_preview = tok.decode(outputs[0], skip_special_tokens=True)[-200:]
        except:
            full_output_preview = "Could not decode"
        
        debug_info = f"""**Warning:** Model generated empty or very short response.

**Debug Information:**
- Input prompt length: {len(prompt)} characters
- Generated tokens: {outputs.shape[1] - input_length}
- Model device: {model_device}
- Temperature: {temperature}
- Max new tokens: {max_new_tokens}

**Possible causes:**
1. Prompt format not compatible with model
2. Model may need different generation parameters
3. Input too long or truncated

**Try:**
- Adjusting temperature settings
- Using a different prompt format
- Checking model documentation for correct format

**Full decoded output (last 200 chars):** {full_output_preview}
"""
        return debug_info
    
    print(f"📝 Response generated: {len(response)} characters")
    print(f"📄 Response preview: {response[:100]}...")
    
    return response


def _call_debug_echo(messages: List[Dict]) -> str:
    """Debug mode that echoes back the last user message."""
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = m.get("content", "")
            break
    return "DEBUG (no provider configured): " + last_user[:600]
