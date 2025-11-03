"""
HuggingFace Inference API backend with proper chat template handling.
"""
import os
import json
import warnings
import requests
from transformers import AutoTokenizer
from typing import List, Dict

# Get HF token from environment, with optional hardcoded fallback (NOT RECOMMENDED for production)
# You can set this to your token if you want, but it's better to use environment variables
_HARDCODED_TOKEN = None  # Set this to "hf_..." if you want to hardcode (not recommended)

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or _HARDCODED_TOKEN
if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN environment variable not set. Please set it:\n"
        "  PowerShell: $env:HF_TOKEN='your-token'\n"
        "  Linux/Mac:  export HF_TOKEN='your-token'\n"
        "Or set _HARDCODED_TOKEN in backend_hf.py (not recommended for production)"
    )

TIMEOUT_S = 180

FINETUNED_MODELS = {
    "MedLlama-HF-API": {
        "description": "MedLlama - Hugging Face Inference API",
        "specialization": "Clinical reasoning and medical decision making",
        "api_url": "https://router.huggingface.co/hf-inference/models/AparnaSuresh/MedLlama-3b?wait_for_model=true",
        "provider": "HuggingFace",
        "is_finetuned": True,
        "use_api": True,
        "use_hf_inference": True
    }
}

# Default Llama chat template (if model doesn't have one)
DEFAULT_LLAMA_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% elif message['role'] == 'user' %}{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% elif message['role'] == 'assistant' %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

# Load tokenizer once (for chat template)
TOKENIZER_ID = "AparnaSuresh/MedLlama-3b"
_tokenizer = None
_has_chat_template = False

try:
    # Suppress warnings during tokenizer loading
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_ID, 
            use_fast=True,
            token=HF_TOKEN  # In case repo is private
        )
    # Check if tokenizer has a chat template
    if _tokenizer and hasattr(_tokenizer, "chat_template") and _tokenizer.chat_template is not None:
        _has_chat_template = True
except Exception:
    # Silent fallback - tokenizer not critical for API calls
    _tokenizer = None
    _has_chat_template = False


def to_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Convert messages to prompt using the model's chat template.
    Falls back to Llama-style format if tokenizer/template not available.
    """
    # Try using tokenizer's chat template if available
    if _tokenizer and _has_chat_template:
        try:
            # Suppress warnings about missing chat template
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return _tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
        except Exception:
            # Silent fallback - template might not work
            pass
    
    # Fallback: Llama-style format (works well for most models)
    prompt_parts = []
    system_content = None
    
    # Extract system message if present
    for msg in messages:
        if msg.get("role") == "system":
            system_content = msg.get("content", "")
            break
    
    # Format messages
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "system":
            # System message will be included at the start
            continue
        elif role == "user":
            prompt_parts.append(f"### User:\n{content}\n\n")
        elif role == "assistant":
            prompt_parts.append(f"### Assistant:\n{content}\n\n")
    
    # Add system message at the beginning if present
    full_prompt = ""
    if system_content:
        full_prompt = f"### System:\n{system_content}\n\n"
    
    full_prompt += "".join(prompt_parts) + "### Assistant:\n"
    
    return full_prompt


def call_hf_model(
    model_key: str, 
    messages: List[Dict[str, str]], 
    max_new_tokens: int = 512, 
    temperature: float = 0.2
) -> str:
    """
    Call HuggingFace Inference API model.
    
    Args:
        model_key: Key in FINETUNED_MODELS
        messages: List of message dicts with 'role' and 'content'
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Generated text response
    """
    if model_key not in FINETUNED_MODELS:
        raise ValueError(f"Unknown model: {model_key}")
    
    cfg = FINETUNED_MODELS[model_key]
    url = cfg["api_url"]
    
    # Convert messages to prompt using chat template
    prompt = to_prompt(messages)
    
    # HF router expects JSON with "inputs" and optional "parameters"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "return_full_text": False,
        }
    }
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    
    try:
        r = requests.post(
            url, 
            headers=headers, 
            json=payload,  # Use json= instead of data=json.dumps()
            timeout=TIMEOUT_S
        )
        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if r.status_code == 404:
            raise ValueError(f"Model not found at {url}. Check if the repository exists and has model files.")
        elif r.status_code == 401:
            raise ValueError("Authentication failed. Check your HF_TOKEN.")
        else:
            raise ValueError(f"API error ({r.status_code}): {r.text}")
    except requests.exceptions.Timeout:
        raise ValueError(f"Request timed out after {TIMEOUT_S} seconds. The model may be loading.")
    
    # Serverless responses vary: either pure string or {"generated_text": "..."} or a list of dicts
    try:
        data = r.json()
    except json.JSONDecodeError:
        # Sometimes HF returns plain text
        return r.text.strip()
    
    # Extract generated text from various response formats
    if isinstance(data, str):
        return data
    
    if isinstance(data, dict):
        if "generated_text" in data:
            return data["generated_text"]
        if "text" in data:
            return data["text"]
        if "output" in data:
            return data["output"]
    
    if isinstance(data, list) and len(data) > 0:
        first = data[0]
        if isinstance(first, dict):
            if "generated_text" in first:
                return first["generated_text"]
            if "text" in first:
                return first["text"]
        elif isinstance(first, str):
            return first
    
    # Fallback: stringify the response
    return str(data)

