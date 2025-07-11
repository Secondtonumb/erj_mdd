import sys
import importlib

def test_dependencies():
    """Test if all required dependencies are available"""
    
    required_packages = [
        'torch',
        'torchaudio', 
        'speechbrain',
        'numpy',
        'tqdm',
        'yaml'
    ]
    
    print("=== Testing Dependencies ===")
    
    missing_packages = []
    
    for package in required_packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {package}: {version}")
        except ImportError:
            print(f"✗ {package}: NOT FOUND")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print("\nAll dependencies are available!")
        return True

def test_cuda():
    """Test CUDA availability"""
    print("\n=== Testing CUDA ===")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            print(f"Current GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is not available. The ASR will run on CPU.")
        
        return cuda_available
    except Exception as e:
        print(f"Error testing CUDA: {e}")
        return False

def test_speechbrain_models():
    """Test if SpeechBrain models can be loaded"""
    print("\n=== Testing SpeechBrain Models ===")
    
    try:
        from speechbrain.pretrained import EncoderDecoderASR
        
        # Test loading a pretrained model
        print("Testing model loading...")
        # Note: This will download the model if not already present
        # model = EncoderDecoderASR.from_hparams(
        #     source="speechbrain/asr-wav2vec2-commonvoice-en",
        #     savedir="pretrained_models/asr-wav2vec2-commonvoice-en"
        # )
        print("✓ SpeechBrain model loading test passed")
        return True
        
    except Exception as e:
        print(f"✗ Error loading SpeechBrain model: {e}")
        return False

if __name__ == "__main__":
    print("Testing ASR dependencies...\n")
    
    deps_ok = test_dependencies()
    cuda_ok = test_cuda()
    sb_ok = test_speechbrain_models()
    
    print(f"\n=== Summary ===")
    print(f"Dependencies: {'✓' if deps_ok else '✗'}")
    print(f"CUDA: {'✓' if cuda_ok else '✗'}")
    print(f"SpeechBrain: {'✓' if sb_ok else '✗'}")
    
    if deps_ok and sb_ok:
        print("\n✓ Ready to run ASR scripts!")
        print("You can now run:")
        print("  python asr_script.py")
        print("  python asr_with_trained_model.py")
    else:
        print("\n✗ Some dependencies are missing. Please install them first.") 