#!/usr/bin/env python3
"""
Script to install and verify OpenAI Whisper installation
"""

import subprocess
import sys
import importlib

def install_whisper():
    """Install OpenAI Whisper package"""
    print("Installing OpenAI Whisper...")
    
    try:
        # Try to install openai-whisper
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openai-whisper"])
        print("✅ Successfully installed openai-whisper")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install openai-whisper via pip")
        
        try:
            # Try alternative installation method
            print("Trying alternative installation method...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "git+https://github.com/openai/whisper.git"
            ])
            print("✅ Successfully installed OpenAI Whisper from GitHub")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install from GitHub")
            return False

def check_whisper_installation():
    """Check if Whisper is properly installed"""
    print("\nChecking Whisper installation...")
    
    try:
        import whisper
        
        # Check if it has the required functions
        if hasattr(whisper, 'load_model'):
            print("✅ OpenAI Whisper is properly installed")
            print(f"   Whisper version: {getattr(whisper, '__version__', 'unknown')}")
            return True
        else:
            print("❌ Wrong whisper package installed")
            print("   This might be a different 'whisper' package")
            return False
            
    except ImportError:
        print("❌ Whisper not found")
        return False

def test_whisper_model():
    """Test loading a small Whisper model"""
    print("\nTesting Whisper model loading...")
    
    try:
        import whisper
        import torch
        
        # Check CUDA availability
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   Using device: {device}")
        
        # Try to load tiny model (fastest to test)
        print("   Loading tiny model for testing...")
        model = whisper.load_model("tiny", device=device)
        print("✅ Whisper model loaded successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load Whisper model: {e}")
        return False

def main():
    print("=== OpenAI Whisper Installation Checker ===\n")
    
    # Check current installation
    if check_whisper_installation():
        if test_whisper_model():
            print("\n🎉 Whisper is ready to use!")
            print("You can now run:")
            print("  python whisper_asr_fixed.py")
            return
    
    # If not properly installed, try to install
    print("\nWhisper is not properly installed. Attempting installation...")
    
    if install_whisper():
        if check_whisper_installation() and test_whisper_model():
            print("\n🎉 Whisper installation completed successfully!")
            print("You can now run:")
            print("  python whisper_asr_fixed.py")
        else:
            print("\n❌ Installation completed but verification failed")
            print("Please try manually:")
            print("  pip uninstall whisper")
            print("  pip install openai-whisper")
    else:
        print("\n❌ Failed to install Whisper")
        print("Please install manually:")
        print("  pip install openai-whisper")

if __name__ == "__main__":
    main() 