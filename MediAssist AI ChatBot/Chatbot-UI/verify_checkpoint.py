"""
Quick script to verify your BioMistral checkpoint structure.
Run this before starting the chatbot to ensure everything is set up correctly.
"""

import os
import sys

def verify_checkpoint(checkpoint_path):
    """Verify checkpoint has required files."""
    print("=" * 60)
    print("BioMistral Checkpoint Verification")
    print("=" * 60)
    print(f"\nChecking: {checkpoint_path}\n")
    
    # Check if directory exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ ERROR: Checkpoint directory not found!")
        print(f"   Expected location: {os.path.abspath(checkpoint_path)}")
        print(f"\n💡 Solutions:")
        print(f"   1. Create the directory: mkdir {checkpoint_path}")
        print(f"   2. Move your checkpoint to this location")
        print(f"   3. Update path in src/config/constants.py")
        return False
    
    print(f"✅ Checkpoint directory found: {os.path.abspath(checkpoint_path)}\n")
    
    # Required files
    required_files = {
        "config.json": "Model configuration",
        "tokenizer_config.json": "Tokenizer configuration",
    }
    
    # Model weight files (at least one required)
    weight_files = {
        "pytorch_model.bin": "PyTorch model weights",
        "model.safetensors": "SafeTensors model weights (preferred)",
    }
    
    # Optional files
    optional_files = {
        "tokenizer.json": "Tokenizer vocabulary",
        "special_tokens_map.json": "Special tokens mapping",
        "generation_config.json": "Generation configuration",
        "training_args.bin": "Training arguments",
    }
    
    all_good = True
    
    # Check required files
    print("Required Files:")
    print("-" * 60)
    for file, description in required_files.items():
        path = os.path.join(checkpoint_path, file)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"✅ {file:<30} ({size:,} bytes)")
        else:
            print(f"❌ {file:<30} MISSING - {description}")
            all_good = False
    
    # Check weight files
    print("\nModel Weights (need at least one):")
    print("-" * 60)
    has_weights = False
    for file, description in weight_files.items():
        path = os.path.join(checkpoint_path, file)
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024**3)  # Convert to GB
            print(f"✅ {file:<30} ({size:.2f} GB)")
            has_weights = True
        else:
            print(f"⚠️  {file:<30} Not found")
    
    if not has_weights:
        print(f"\n❌ ERROR: No model weight files found!")
        all_good = False
    
    # Check optional files
    print("\nOptional Files:")
    print("-" * 60)
    for file, description in optional_files.items():
        path = os.path.join(checkpoint_path, file)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"✅ {file:<30} ({size:,} bytes)")
        else:
            print(f"⚠️  {file:<30} Not found (optional)")
    
    # Check for sharded models
    print("\nChecking for sharded model files...")
    print("-" * 60)
    sharded_files = [f for f in os.listdir(checkpoint_path) if 'pytorch_model' in f and 'of' in f]
    if sharded_files:
        print(f"✅ Found {len(sharded_files)} sharded model files:")
        for f in sorted(sharded_files):
            size = os.path.getsize(os.path.join(checkpoint_path, f)) / (1024**3)
            print(f"   - {f} ({size:.2f} GB)")
    else:
        print("⚠️  No sharded files (this is okay for smaller models)")
    
    # Summary
    print("\n" + "=" * 60)
    if all_good:
        print("✅ SUCCESS! Your checkpoint is properly structured.")
        print("\n🚀 You can now run: streamlit run app.py")
        print("\n💡 In the app:")
        print("   1. Select 'BioMistral-7B' from the model dropdown")
        print("   2. Choose a medical department")
        print("   3. Start asking questions!")
    else:
        print("❌ ISSUES FOUND! Please fix the problems above.")
        print("\n💡 Common solutions:")
        print("   1. Ensure all required files are present")
        print("   2. Check file permissions")
        print("   3. Verify checkpoint was saved correctly during training")
    print("=" * 60)
    
    return all_good


if __name__ == "__main__":
    # Default checkpoint path (matches constants.py)
    checkpoint_path = "./checkpoint-biomistral"
    
    # Allow custom path as command line argument
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    
    verify_checkpoint(checkpoint_path)


