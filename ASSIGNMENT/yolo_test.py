"""
PHASE 1 - YOLO MODEL VERIFICATION

DSA02 - Computer Vision with OpenCV
Autonomous Road Safety System

Purpose:
    Verify that the Ultralytics YOLO framework is installed
    correctly and that a pretrained YOLO model can be loaded.
"""

from ultralytics import YOLO


def main():

    print("=" * 65)
    print("AUTONOMOUS ROAD SAFETY SYSTEM")
    print("DSA02 - COMPUTER VISION WITH OPENCV")
    print("PHASE 1 - YOLO MODEL VERIFICATION")
    print("=" * 65)

    print()
    print("Step 1: Loading YOLO model...")
    
    try:
        model = YOLO("yolo11n.pt")

        print("Step 2: YOLO model loaded successfully.")
        print()

        print("-" * 65)
        print("MODEL INFORMATION")
        print("-" * 65)

        print("Model Type :", type(model).__name__)
        print("Model File : yolo11n.pt")

        print()
        print("-" * 65)
        print("YOLO CLASS NAMES")
        print("-" * 65)

        if hasattr(model, "names"):
            for class_id, class_name in model.names.items():
                print(f"{class_id:3d} : {class_name}")

        print()
        print("-" * 65)
        print("PHASE 1 YOLO TEST : PASSED")
        print("-" * 65)

    except Exception as error:

        print()
        print("-" * 65)
        print("YOLO MODEL TEST : FAILED")
        print("-" * 65)

        print("Error:")
        print(error)

    print()
    print("=" * 65)


if __name__ == "__main__":
    main()
