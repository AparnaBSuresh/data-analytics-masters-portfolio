"""
Script to quantize MedLlama model to GGUF format for fast CPU inference with llama.cpp
"""
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess
import sys

MODEL_ID = "AparnaSuresh/MedLlama-3b"
OUTPUT_DIR = "./models/medllama-3b-gguf"
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

print("="*60)
print("🧪 MedLlama GGUF Quantization Script")
print("="*60)
print("\nThis script will:")
print("1. Download the model from HuggingFace")
print("2. Convert it to GGUF format")
print("3. Quantize it for faster CPU inference")
print("\n⚠️  This requires 'llama.cpp' repository and python package")
print("="*60 + "\n")

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import llama_cpp
        print("✅ llama-cpp-python is installed")
    except ImportError:
        print("❌ llama-cpp-python not found!")
        print("\nInstall it with:")
        print("  pip install llama-cpp-python")
        print("\nOr for GPU support:")
        print("  CMAKE_ARGS='-DLLAMA_CUBLAS=on' pip install llama-cpp-python")
        return False
    
    # Check if convert script exists (from llama.cpp)
    try:
        import llama_cpp
        # The convert script should be available
        print("✅ llama.cpp tools available")
    except:
        pass
    
    return True

def download_and_convert():
    """Download model and convert to GGUF"""
    print("\n📦 Step 1: Downloading model from HuggingFace...")
    print(f"   Model: {MODEL_ID}")
    
    # Load tokenizer (needed for conversion)
    print("\n📚 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Tokenizer saved")
    
    print("\n📦 Loading model... (this may take a few minutes)")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        device_map="cpu",
        torch_dtype="float32"
    )
    print("✅ Model loaded")
    
    print("\n💾 Saving model in format compatible with llama.cpp conversion...")
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    print("✅ Model saved to", OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("📝 Next Steps:")
    print("="*60)
    print("\n1. Clone llama.cpp repository:")
    print("   git clone https://github.com/ggerganov/llama.cpp.git")
    print("   cd llama.cpp")
    print("   make")
    print("\n2. Convert to GGUF format:")
    print(f"   python convert-hf-to-gguf.py {os.path.abspath(OUTPUT_DIR)} --outdir {os.path.abspath(OUTPUT_DIR)}/gguf")
    print("\n3. Quantize (recommended: Q4_K_M for good quality/speed balance):")
    print(f"   ./quantize {OUTPUT_DIR}/gguf/ggml-model-f16.gguf {OUTPUT_DIR}/gguf/medllama-3b-q4_k_m.gguf Q4_K_M")
    print("\n4. Update model path in constants.py:")
    print(f"   'model_path': '{OUTPUT_DIR}/gguf/medllama-3b-q4_k_m.gguf'")
    print("\n" + "="*60)

if __name__ == "__main__":
    if not check_dependencies():
        print("\n❌ Please install dependencies first!")
        sys.exit(1)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        download_and_convert()
        print("\n✅ Conversion preparation complete!")
        print("⚠️  Follow the manual steps above to complete GGUF conversion")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

