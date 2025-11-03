"""
Clear the GGUF model cache to force reload.
This is useful if you switched from F16 to Q4_K_M model.
"""
import sys

# Clear the cache in backend_gguf module
try:
    from src.llm.backend_gguf import _gguf_model_cache
    _gguf_model_cache.clear()
    print("✅ GGUF model cache cleared!")
    print("   The app will reload the quantized model on next run.")
except Exception as e:
    print(f"❌ Error: {e}")

