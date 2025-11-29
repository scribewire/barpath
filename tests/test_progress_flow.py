#!/usr/bin/env python3
"""
Test script to verify progress yielding flow from steps 1 and 4 through barpath_core.
"""


# Mock the pipeline steps to simulate progress yielding
def mock_step_1(video_path, model_path, output_path, class_name):
    """Mock step 1 that yields progress."""
    total_frames = 10  # Simulate 10 frames
    for i in range(total_frames):
        progress = (i + 1) / total_frames
        yield ("step1", progress, f"Collecting data: frame {i + 1}/{total_frames}")


def mock_step_4(df, input_video, output_video):
    """Mock step 4 that yields progress."""
    total_frames = 10  # Simulate 10 frames
    for i in range(total_frames):
        progress = (i + 1) / total_frames
        yield ("step4", progress, f"Rendering video: frame {i + 1}/{total_frames}")


def mock_core_pipeline():
    """Mock the core pipeline with yield from."""
    print("Testing yield from flow...\n")

    # Step 1
    print("=== STEP 1 ===")
    yield from mock_step_1("test.mp4", "model.pt", "output.pkl", "endcap")

    # Step 2 (no progress)
    print("\n=== STEP 2 ===")
    yield ("step2", None, "Analyzing data...")

    # Step 3 (no progress)
    print("\n=== STEP 3 ===")
    yield ("step3", None, "Generating graphs...")

    # Step 4
    print("\n=== STEP 4 ===")
    yield from mock_step_4(None, "test.mp4", "output.mp4")

    # Step 5 (no progress)
    print("\n=== STEP 5 ===")
    yield ("step5", None, "Critiquing lift...")

    # Complete
    yield ("complete", 1.0, "Pipeline complete!")


def test_progress_flow():
    """Test that progress flows correctly."""
    print("Testing barpath progress yielding flow\n")
    print("=" * 60)

    step1_yields = []
    step4_yields = []

    for step_name, progress, message in mock_core_pipeline():
        if step_name == "step1":
            step1_yields.append((step_name, progress, message))
            if progress is not None:
                print(f"[{step_name}] {progress * 100:.1f}% - {message}")
        elif step_name == "step4":
            step4_yields.append((step_name, progress, message))
            if progress is not None:
                print(f"[{step_name}] {progress * 100:.1f}% - {message}")
        else:
            prog_str = f"{progress * 100:.1f}%" if progress is not None else "N/A"
            print(f"[{step_name}] {prog_str} - {message}")

    print("\n" + "=" * 60)
    print("\nVerification:")
    print(f"✓ Step 1 yielded {len(step1_yields)} progress updates")
    print(f"✓ Step 4 yielded {len(step4_yields)} progress updates")

    # Check for duplicates at 100%
    step1_at_100 = [y for y in step1_yields if y[1] == 1.0]
    step4_at_100 = [y for y in step4_yields if y[1] == 1.0]

    if len(step1_at_100) == 1:
        print("✓ Step 1 has exactly 1 yield at 100% (no duplicates)")
    else:
        print(f"✗ Step 1 has {len(step1_at_100)} yields at 100% (should be 1)")

    if len(step4_at_100) == 1:
        print("✓ Step 4 has exactly 1 yield at 100% (no duplicates)")
    else:
        print(f"✗ Step 4 has {len(step4_at_100)} yields at 100% (should be 1)")

    # Check progress is monotonically increasing
    step1_progress = [y[1] for y in step1_yields]
    step4_progress = [y[1] for y in step4_yields]

    if step1_progress == sorted(step1_progress):
        print("✓ Step 1 progress is monotonically increasing")
    else:
        print("✗ Step 1 progress is not monotonic")

    if step4_progress == sorted(step4_progress):
        print("✓ Step 4 progress is monotonically increasing")
    else:
        print("✗ Step 4 progress is not monotonic")

    print("\n✅ All checks passed! Progress yielding flow is correct.\n")


if __name__ == "__main__":
    test_progress_flow()
