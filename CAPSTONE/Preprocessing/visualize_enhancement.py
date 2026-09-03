import os
import cv2
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ORIGINAL_DATASET = r"C:\Users\veman\OneDrive\Desktop\Open CV New Slot\fish_image"

ENHANCED_DATASET = r"C:\DeepFishVision\outputs\enhanced_images"

OUTPUT_FOLDER = r"C:\DeepFishVision\outputs\graphs"


# ============================================================
# FIND FIRST IMAGE
# ============================================================

def find_first_image(folder):

    image_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    )

    for root, dirs, files in os.walk(folder):

        for filename in files:

            if filename.lower().endswith(image_extensions):

                return os.path.join(
                    root,
                    filename
                )

    return None


# ============================================================
# VISUALIZE ORIGINAL AND ENHANCED IMAGE
# ============================================================

def visualize():

    print("=" * 60)
    print("DeepFishVision - OpenCV Enhancement Visualization")
    print("=" * 60)

    # Find original image
    original_path = find_first_image(
        ORIGINAL_DATASET
    )

    if original_path is None:

        print("No image found in original dataset.")
        return

    # Find corresponding enhanced image
    relative_path = os.path.relpath(
        original_path,
        ORIGINAL_DATASET
    )

    enhanced_path = os.path.join(
        ENHANCED_DATASET,
        relative_path
    )

    # Read original image
    original = cv2.imread(
        original_path
    )

    # Read enhanced image
    enhanced = cv2.imread(
        enhanced_path
    )

    if original is None:

        print("Unable to read original image.")
        return

    if enhanced is None:

        print("Enhanced image not found.")
        print(enhanced_path)
        return

    # Convert BGR to RGB
    original_rgb = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )

    enhanced_rgb = cv2.cvtColor(
        enhanced,
        cv2.COLOR_BGR2RGB
    )

    # Create output directory
    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    # ========================================================
    # CREATE COMPARISON
    # ========================================================

    plt.figure(
        figsize=(12, 6)
    )

    # Original image
    plt.subplot(
        1,
        2,
        1
    )

    plt.imshow(
        original_rgb
    )

    plt.title(
        "Original Underwater Image"
    )

    plt.axis(
        "off"
    )

    # Enhanced image
    plt.subplot(
        1,
        2,
        2
    )

    plt.imshow(
        enhanced_rgb
    )

    plt.title(
        "OpenCV Enhanced Image"
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    # Save comparison
    output_file = os.path.join(
        OUTPUT_FOLDER,
        "original_vs_enhanced.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("\nComparison saved successfully!")
    print(output_file)


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    visualize()