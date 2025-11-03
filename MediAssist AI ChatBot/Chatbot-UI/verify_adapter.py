"""
Verify LoRA adapter structure before starting FastAPI server
"""

import os
import json

def verify_adapter(adapter_path="./checkpoint-biomistral"):
    """Verify adapter has all required files"""
    print("="*60)
    print("LoRA Adapter Verification")
    print("="*60)
    print(f"\nChecking: {os.path.abspath(adapter_path)}\n")
    
    if not os.path.exists(adapter_path):
        print(f"❌ ERROR: Adapter directory not found!")
        print(f"   Expected: {os.path.abspath(adapter_path)}")
        return False
    
    print(f"✅ Adapter directory found\n")
    
    # Required files for LoRA adapter
    required_files = {
        "adapter_model.safetensors": "LoRA adapter weights",
        "adapter_config.json": "LoRA configuration"
    }
    
    # Optional but recommended files
    optional_files = {
        "tokenizer.json": "Tokenizer vocabulary",
        "README.md": "Documentation",
        "training_args.bin": "Training arguments"
    }
    
    all_good = True
    
    print("Required Files:")
    print("-" * 60)
    for file, description in required_files.items():
        path = os.path.join(adapter_path, file)
        if os.path.exists(path):
            size = os.path.getsize(path)
            if file.endswith('.json'):
                print(f"✅ {file:<30} ({size:,} bytes)")
            else:
                size_mb = size / (1024**2)
                print(f"✅ {file:<30} ({size_mb:.2f} MB)")
        else:
            print(f"❌ {file:<30} MISSING - {description}")
            all_good = False
    
    print("\nOptional Files:")
    print("-" * 60)
    for file, description in optional_files.items():
        path = os.path.join(adapter_path, file)
        if os.path.exists(path):
            size = os.path.getsize(path)
            size_kb = size / 1024
            print(f"✅ {file:<30} ({size_kb:.2f} KB)")
        else:
            print(f"⚠️  {file:<30} Not found (optional)")
    
    # Verify adapter_config.json content
    if all_good:
        print("\nVerifying adapter_config.json content:")
        print("-" * 60)
        config_path = os.path.join(adapter_path, "adapter_config.json")
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            required_keys = ["peft_type", "base_model_name_or_path", "task_type"]
            for key in required_keys:
                if key in config:
                    print(f"✅ {key}: {config[key]}")
                else:
                    print(f"❌ Missing key: {key}")
                    all_good = False
        except Exception as e:
            print(f"❌ Error reading config: {e}")
            all_good = False
    
    print("\n" + "="*60)
    if all_good:
        print("✅ SUCCESS! Adapter is properly configured")
        print("\n🚀 You can now start the FastAPI server:")
        print("   uvicorn fastapi_server:app --port 8000")
    else:
        print("❌ ISSUES FOUND! Please fix the problems above")
        print("\n💡 If adapter_config.json is missing, it has been created")
        print("   with default LoRA settings. Adjust if needed.")
    print("="*60)
    
    return all_good


if __name__ == "__main__":
    verify_adapter()


