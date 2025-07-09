import torch
import sys

def test_cuda():
    print("=== CUDA Availability Test ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        # Get CUDA device count
        device_count = torch.cuda.device_count()
        print(f"CUDA device count: {device_count}")
        
        # Get current device
        current_device = torch.cuda.current_device()
        print(f"Current CUDA device: {current_device}")
        
        # Get device name
        device_name = torch.cuda.get_device_name(current_device)
        print(f"Device name: {device_name}")
        
        # Get CUDA version
        cuda_version = torch.version.cuda
        print(f"CUDA version: {cuda_version}")
        
        # Test tensor operations on GPU
        print("\n=== Testing GPU Operations ===")
        try:
            # Create a tensor on GPU
            x = torch.randn(3, 3).cuda()
            y = torch.randn(3, 3).cuda()
            
            # Perform operations
            z = torch.matmul(x, y)
            print(f"GPU tensor shape: {z.shape}")
            print(f"GPU tensor device: {z.device}")
            print("GPU operations successful!")
            
        except Exception as e:
            print(f"Error during GPU operations: {e}")
    else:
        print("CUDA is not available. This could be due to:")
        print("1. No NVIDIA GPU installed")
        print("2. CUDA drivers not installed")
        print("3. PyTorch not compiled with CUDA support")
        print("4. CUDA version mismatch")
    
    # Test CPU operations
    print("\n=== Testing CPU Operations ===")
    try:
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = torch.matmul(x, y)
        print(f"CPU tensor shape: {z.shape}")
        print(f"CPU tensor device: {z.device}")
        print("CPU operations successful!")
    except Exception as e:
        print(f"Error during CPU operations: {e}")

if __name__ == "__main__":
    test_cuda() 