"""
GGUF model backend using llama-cpp-python for fast CPU inference.
This is much faster than transformers for CPU inference.
"""
import os
import time
from typing import List, Dict, Optional, Tuple

# Global cache for loaded GGUF models
# NOTE: This cache is keyed by model_path, so different model files will be cached separately
_gguf_model_cache = {}

def load_gguf_model(model_path: str, n_ctx: int = 1024, n_threads: Optional[int] = None, n_gpu_layers: int = 0):
    """
    Load a GGUF model using llama-cpp-python.
    
    Args:
        model_path: Path to .gguf file
        n_ctx: Context window size
        n_threads: Number of CPU threads (None = auto)
        n_gpu_layers: Number of layers to offload to GPU (0 = CPU only)
    
    Returns:
        Loaded Llama model
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError(
            "llama-cpp-python not installed. Install it with:\n"
            "  pip install llama-cpp-python\n"
            "\nFor GPU support:\n"
            "  CMAKE_ARGS='-DLLAMA_CUBLAS=on' pip install llama-cpp-python"
        )
    
    # Check cache first
    cache_key = f"{model_path}_{n_ctx}_{n_threads}_{n_gpu_layers}"
    if cache_key in _gguf_model_cache:
        return _gguf_model_cache[cache_key]
    
    print(f"🔄 Loading GGUF model from: {model_path}")
    
    # Check if it's quantized model
    if "q4_k_m" in model_path.lower() or "q4" in model_path.lower():
        print("✅ Using QUANTIZED model (Q4_K_M) - Should be 5-10x faster!")
    elif "f16" in model_path.lower() or "f32" in model_path.lower():
        print("⚠️ Using F16/F32 model - This is SLOW on CPU. Consider quantizing to Q4_K_M.")
    
    # Auto-detect number of threads if not specified
    # Optimize: Use more threads for faster inference (but leave 1 core free)
    if n_threads is None:
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        # Use most cores, but leave 1 free for system
        n_threads = max(1, cpu_count - 1)
        print(f"📊 Using {n_threads} CPU threads (auto-detected from {cpu_count} cores)")
    
    try:
        model = Llama(
            model_path=model_path,
            n_ctx=min(n_ctx, 512),  # Smaller context for faster processing
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            use_mlock=False,  # Disable for faster access (if you have enough RAM)
            n_batch=256,  # Much smaller batch for faster CPU processing
            f16_kv=True,  # Use FP16 for key-value cache (faster)
            use_mmap=True,  # Memory mapping for faster loading
            n_parts=1,  # Single part for faster loading
        )
        
        # Cache the model
        _gguf_model_cache[cache_key] = model
        print("✅ GGUF model loaded and cached!")
        
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load GGUF model from {model_path}: {str(e)}")


def call_gguf_model(
    messages: List[Dict[str, str]],
    model_path: str,
    max_tokens: int = 128,
    temperature: float = 0.2,
    n_ctx: int = 1024,
    n_threads: Optional[int] = None,
    n_gpu_layers: int = 0
) -> Tuple[str, Dict[str, float]]:
    """
    Generate response using GGUF model.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        model_path: Path to .gguf file
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        n_ctx: Context window size
        n_threads: Number of CPU threads
        n_gpu_layers: GPU layers (0 = CPU only)
    
    Returns:
        Generated text response
    """
    # Load model
    model = load_gguf_model(model_path, n_ctx=n_ctx, n_threads=n_threads, n_gpu_layers=n_gpu_layers)
    
    # Format messages for llama.cpp - clean, focused prompt
    # Extract only the current user question (last user message)
    system_content = None
    user_content = None
    current_user_msg = None
    
    # Find the most recent user message (this is what we want to answer)
    for msg in reversed(messages):
        if msg.get("role") == "user":
            current_user_msg = msg.get("content", "")
            break
    
    # If no user message found, use the last message
    if not current_user_msg and messages:
        current_user_msg = messages[-1].get("content", "")
        user_content = current_user_msg
    else:
        user_content = current_user_msg
    
    # Get system prompt (first system message, truncated)
    for msg in messages:
        if msg.get("role") == "system":
            system_content = msg.get("content", "")[:200] if len(msg.get("content", "")) > 200 else msg.get("content", "")
            break
    
    # Build clean, focused prompt - ONLY answer the current question
    # Use clear format that prevents the model from continuing previous conversations
    if system_content:
        prompt = f"""You are a medical assistant. Answer the following question clearly and concisely.

Question: {user_content}

Answer:"""
    else:
        prompt = f"""Question: {user_content}

Answer:"""
    
    # Start timing
    total_start = time.time()
    prompt_time = time.time()
    
    print(f"🔤 Generating response (prompt length: {len(prompt)} chars)...")
    
    try:
        # Optimize generation parameters for speed
        generation_start = time.time()
        
        # Generate response with optimized settings for speed
        # For Q4_K_M quantized models: Limit tokens for faster responses (even quantized models are slow on CPU with many tokens)
        # For F16 models: Use much smaller max_tokens (they're VERY slow on CPU)
        is_quantized = "q4" in model_path.lower() or "q5" in model_path.lower() or "q6" in model_path.lower() or "q8" in model_path.lower()
        if is_quantized:
            # Limit to 128 tokens for much faster responses (can generate in 15-30s)
            optimized_max_tokens = min(max_tokens, 128)
            print(f"⚡ Using quantized model - generating up to {optimized_max_tokens} tokens for faster responses")
        else:
            optimized_max_tokens = min(max_tokens, 128)  # Cap at 128 for F16 (very slow otherwise)
            print("⚠️ Using F16 model - limiting tokens for speed")
        
        # Optimization: Reduce context window for speed
        # Smaller context = faster processing
        optimized_n_ctx = min(n_ctx, 512)  # Smaller context window for faster processing
        
        # Use only supported parameters for llama-cpp-python
        # Aggressively optimized for speed: minimal sampling, early stopping
        response = model(
            prompt,
            max_tokens=optimized_max_tokens,
            temperature=max(temperature, 0.4),  # Higher temp = faster generation
            stop=["\n\nQuestion:", "\nQuestion:", "Question:", "User:", "System:", "\n\n\n", "###", "\n\nUser", "\n\nSystem", "\n\n", "##", "\nU:", "\nS:"],  # Stop sequences to prevent continuation
            echo=False,  # Don't echo the prompt
            repeat_penalty=1.2,  # Higher to prevent repetition (allows shorter responses)
            top_p=0.9,  # Lower = faster generation
            top_k=30,   # Much lower = fewer candidates = much faster generation
            tfs_z=1.0,  # Tail-free sampling (if supported) - helps speed
        )
        
        generation_time = time.time() - generation_start
        
        # Extract text from response - llama-cpp-python returns Completion object
        extraction_start = time.time()
        
        try:
            # Try dict access first (most common)
            if isinstance(response, dict):
                generated_text = response.get("choices", [{}])[0].get("text", "")
                usage = response.get("usage", {})
            # Try response.choices[0].text (llama-cpp-python object)
            elif hasattr(response, 'choices') and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice, 'text'):
                    generated_text = choice.text
                elif isinstance(choice, dict):
                    generated_text = choice.get('text', '')
                else:
                    generated_text = str(choice)
                # Get usage if available
                if hasattr(response, 'usage'):
                    usage = response.usage if isinstance(response.usage, dict) else {}
                else:
                    usage = {}
            # Try dict-style access on object
            elif hasattr(response, '__getitem__'):
                try:
                    generated_text = response['choices'][0]['text']
                    usage = response.get('usage', {})
                except (KeyError, TypeError):
                    generated_text = str(response)
                    usage = {}
            else:
                # Last resort: convert to string
                generated_text = str(response)
                usage = {}
            
            generated_text = generated_text.strip() if generated_text else ""
            
            # Clean up the response - remove any continuation patterns
            # Stop at first occurrence of Q&A patterns that indicate model is continuing
            stop_patterns = [
                "\n\nU:", "\nU:", "U: ", "User:", "\nUser:",
                "\n\nQuestion:", "\nQuestion:", "Question: ",
                "\n\nQ:", "\nQ:", "Q: ",
                "###", "\n\n\n"
            ]
            
            # Find the earliest stop pattern and truncate there
            earliest_stop = len(generated_text)
            for pattern in stop_patterns:
                idx = generated_text.find(pattern)
                if idx != -1 and idx < earliest_stop:
                    earliest_stop = idx
            
            if earliest_stop < len(generated_text):
                generated_text = generated_text[:earliest_stop].strip()
                print(f"⚠️ Truncated response at stop pattern (prevented continuation)")
            
            # Remove any trailing incomplete sentences (ending with "U:", "A:", etc.)
            lines = generated_text.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith(('U:', 'A:', 'Question:', 'Answer:', 'User:', 'System:')):
                    break  # Stop at first pattern that looks like continuation
                cleaned_lines.append(line)
            
            generated_text = '\n'.join(cleaned_lines).strip()
            
        except Exception as extract_error:
            print(f"❌ Error extracting response: {extract_error}")
            print(f"   Response type: {type(response)}")
            print(f"   Response: {repr(response)[:500]}")
            generated_text = ""
            usage = {}
        
        print(f"✅ Extracted {len(generated_text)} characters from response")
        extraction_time = time.time() - extraction_start
        
        total_time = time.time() - total_start
        prompt_build_time = prompt_time - total_start
        
        # Calculate tokens per second if available
        tokens_generated = usage.get("completion_tokens", len(generated_text.split()))
        tokens_per_sec = tokens_generated / generation_time if generation_time > 0 else 0
        
        timing_info = {
            "total_time": total_time,
            "prompt_build_time": abs(prompt_build_time),
            "generation_time": generation_time,
            "extraction_time": extraction_time,
            "tokens_generated": tokens_generated,
            "tokens_per_sec": tokens_per_sec,
            "prompt_length": len(prompt),
            "response_length": len(generated_text)
        }
        
        print(f"✅ Generated {len(generated_text)} characters in {total_time:.2f}s ({tokens_per_sec:.1f} tokens/sec)")
        print(f"   ⏱️ Breakdown: Prompt={abs(prompt_build_time):.2f}s, Generate={generation_time:.2f}s, Extract={extraction_time:.2f}s")
        
        if not generated_text or len(generated_text) < 5:
            warning = "**Warning:** Model generated empty or very short response. Please try adjusting temperature or prompt format."
            print(f"⚠️ {warning}")
            print(f"   Generated text length: {len(generated_text)}")
            print(f"   Response object type: {type(response)}")
            print(f"   Response preview: {str(response)[:200]}")
            return (warning, timing_info)
        
        return (generated_text, timing_info)
        
    except Exception as e:
        error_msg = f"Generation error: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        timing_info = {
            "total_time": time.time() - total_start,
            "error": str(e)
        }
        return (f"**Error generating response:** {error_msg}\n\nPlease check the terminal for detailed error information.", timing_info)

