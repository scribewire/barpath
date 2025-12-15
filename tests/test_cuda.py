#!/usr/bin/env python3
"""
CUDA Setup Verification Script for BARPATH

This script tests PyTorch CUDA installation and Ultralytics GPU support.
Run this to verify your NVIDIA GPU acceleration setup.
"""

import sys

import torch


def test_pytorch_cuda():
    """Test basic PyTorch CUDA functionality."""
    print("=== PyTorch CUDA Test ===")

    try:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"GPU count: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

            # Test tensor operations on GPU
            print("\nTesting tensor operations on GPU...")
            device = torch.device("cuda:0")
            x = torch.randn(1000, 1000).to(device)
            y = torch.randn(1000, 1000).to(device)
            z = torch.mm(x, y)
            print(f"Matrix multiplication successful: {z.shape}")

            return True
        else:
            print("CUDA not available. Check your PyTorch installation.")
            return False

    except Exception as e:
        print(f"Error testing PyTorch CUDA: {e}")
        return False


def test_ultralytics_cuda():
    """Test Ultralytics YOLO with CUDA."""
    print("\n=== Ultralytics CUDA Test ===")

    try:
        from ultralytics import YOLO  # type: ignore[attr-defined]

        # Try to load a small model (this will download if not present)
        print("Loading YOLOv8n model...")
        model = YOLO("yolov8n.pt")  # Small model for testing

        # Test inference on CPU first
        print("Testing inference on CPU...")
        results_cpu = model(torch.randn(1, 3, 640, 640), device="cpu", verbose=False)
        # Validate CPU results
        if results_cpu is not None:
            print("CPU inference successful")
        else:
            print("CPU inference failed - no results returned")
            return False

        # Test inference on GPU if available
        if torch.cuda.is_available():
            print("Testing inference on GPU...")
            results_gpu = model(
                torch.randn(1, 3, 640, 640), device="cuda", verbose=False
            )
            # Validate GPU results
            if results_gpu is not None:
                print("GPU inference successful")
            else:
                print("GPU inference failed - no results returned")
                return False
        else:
            print("CUDA not available, skipping GPU inference test")

        return True

    except ImportError:
        print("Ultralytics not installed. Install with: pip install ultralytics")
        return False
    except Exception as e:
        print(f"Error testing Ultralytics CUDA: {e}")
        return False


def main():
    """Main test function."""
    print("BARPATH CUDA Setup Verification")
    print("=" * 40)

    # Test PyTorch CUDA
    pytorch_ok = test_pytorch_cuda()

    # Test Ultralytics CUDA
    ultralytics_ok = test_ultralytics_cuda()

    print("\n" + "=" * 40)
    print("SUMMARY:")
    print(f"PyTorch CUDA: {'✓ PASS' if pytorch_ok else '✗ FAIL'}")
    print(f"Ultralytics CUDA: {'✓ PASS' if ultralytics_ok else '✗ FAIL'}")

    if pytorch_ok and ultralytics_ok:
        print("\n🎉 CUDA setup is working correctly!")
        print("BARPATH should now use GPU acceleration for YOLO inference.")
    else:
        print("\n❌ CUDA setup issues detected.")
        print("Check your PyTorch installation and NVIDIA drivers.")
        print("Installation guide: https://pytorch.org/get-started/locally/")

    return 0 if (pytorch_ok and ultralytics_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
