#!/usr/bin/env python3
"""
GPU Verification Script for M-P6203E Data Projects Hackathon
Run this after GPU setup to verify CUDA is working correctly.
"""

import torch
import sys

def verify_gpu():
    """Verify GPU setup and PyTorch CUDA support."""
    print("=" * 70)
    print("GPU VERIFICATION SCRIPT")
    print("=" * 70)
    
    # Basic info
    print(f"\n✓ PyTorch version: {torch.__version__}")
    print(f"✓ PyTorch built with CUDA: {torch.version.cuda is not None}")
    
    # CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\n{'✓' if cuda_available else '✗'} CUDA available: {cuda_available}")
    
    if not cuda_available:
        print("\n❌ CUDA is NOT available!")
        print("   Please reinstall PyTorch with CUDA support using:")
        print("   conda install -n hack_03 pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia")
        return False
    
    # GPU details
    print(f"\n✓ Number of GPUs: {torch.cuda.device_count()}")
    print(f"✓ Current GPU: {torch.cuda.current_device()}")
    print(f"✓ GPU name: {torch.cuda.get_device_name(0)}")
    
    # GPU properties
    props = torch.cuda.get_device_properties(0)
    print(f"✓ Total GPU memory: {props.total_memory / 1e9:.2f} GB")
    print(f"✓ Compute capability: {props.major}.{props.minor}")
    
    # CUDA version
    print(f"\n✓ CUDA version: {torch.version.cuda}")
    print(f"✓ cuDNN version: {torch.backends.cudnn.version()}")
    
    # Test basic tensor operation
    print("\n" + "-" * 70)
    print("Testing basic GPU tensor operation...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("✓ GPU tensor operation successful!")
        print(f"✓ Result shape: {z.shape}")
    except Exception as e:
        print(f"✗ GPU tensor operation failed: {e}")
        return False
    
    # Benchmark info
    print("\n" + "-" * 70)
    print("Expected Performance Improvements:")
    print("  • Single epoch training: ~10-60 seconds (vs 5-10 minutes on CPU)")
    print(f"  • Batch processing: Accelerated with {torch.cuda.device_count()} GPU(s)")
    
    print("\n" + "=" * 70)
    print("✅ GPU SETUP SUCCESSFUL!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = verify_gpu()
    sys.exit(0 if success else 1)
