"""PyTorch CUDA verification for barpath.

These tests are skipped automatically when CUDA is not available.
"""

import pytest
import torch


def test_pytorch_cuda():
    """Test basic PyTorch CUDA functionality."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        return

    assert torch.cuda.is_available()
    assert torch.version.cuda, "torch.version.cuda is empty"
    assert torch.cuda.device_count() >= 1

    device = torch.device("cuda:0")
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    z = torch.mm(x, y)
    assert z.shape == (1000, 1000)


def test_ultralytics_cuda():
    """Test ultralytics with CUDA using the shipped barbell model."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        return

    from pathlib import Path

    from ultralytics import YOLO  # type: ignore[attr-defined]

    model_path = Path(__file__).parent.parent / "barpath" / "models" / "std_nano.pt"
    if not model_path.exists():
        pytest.skip(f"model not found: {model_path}")
        return

    model = YOLO(str(model_path), task="detect")
    results = model(torch.randn(1, 3, 640, 640), device="cuda:0", verbose=False)
    assert results is not None
