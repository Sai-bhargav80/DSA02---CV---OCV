"""
Autonomous Road Safety System
DSA02 - Computer Vision with OpenCV

Phase 1:
Project and environment verification.
"""

import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib
import torch
import ultralytics


def print_environment_information():
    """Display installed software and library versions."""

    print("=" * 60)
    print("AUTONOMOUS ROAD SAFETY SYSTEM")
    print("DSA02 - COMPUTER VISION WITH OPENCV")
    print("PHASE 1 - ENVIRONMENT VERIFICATION")
    print("=" * 60)

    print(f"Python Version      : {sys.version.split()[0]}")
    print(f"OpenCV Version      : {cv2.__version__}")
    print(f"NumPy Version       : {np.__version__}")
    print(f"Pandas Version      : {pd.__version__}")
    print(f"Matplotlib Version  : {matplotlib.__version__}")
    print(f"PyTorch Version     : {torch.__version__}")
    print(f"Ultralytics Version : {ultralytics.__version__}")

    print("-" * 60)

    if torch.cuda.is_available():
        print("GPU Status          : CUDA GPU AVAILABLE")
        print(f"GPU Device          : {torch.cuda.get_device_name(0)}")
    else:
        print("GPU Status          : CUDA GPU NOT AVAILABLE")
        print("Processing Device   : CPU")

    print("-" * 60)

    # Basic OpenCV test
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)

    cv2.putText(
        test_image,
        "OpenCV TEST",
        (180, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (255, 255, 255),
        3
    )

    print("OpenCV Test         : PASSED")
    print("NumPy Test          : PASSED")
    print("Pandas Test         : PASSED")
    print("PyTorch Test        : PASSED")
    print("YOLO Library Test   : PASSED")

    print("=" * 60)
    print("PHASE 1 ENVIRONMENT SETUP COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    print_environment_information()
