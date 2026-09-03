import os
import cv2


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = r"C:\Users\veman\OneDrive\Desktop\Open CV New Slot\fish_image"

OUTPUT_PATH = r"C:\DeepFishVision\outputs\enhanced_images"


# ============================================================
# IMAGE ENHANCEMENT FUNCTION
# ============================================================

def enhance_image(image):

    # Resize image to 224 x 224
    image = cv2.resize(
        image,
        (224, 224)
    )

    # Noise reduction
    denoised = cv2.GaussianBlur(
        image,
        (3, 3),
        0
    )

    # Convert BGR to LAB
    lab = cv2.cvtColor(
        denoised,
        cv2.COLOR_BGR2LAB
    )

    # Split LAB channels
    l, a, b = cv2.split(lab)

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l)

    # Merge channels
    enhanced_lab = cv2.merge(
        (enhanced_l, a, b)
    )

    # Convert LAB back to BGR
    enhanced_image = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    return enhanced_image


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset():

    # Check dataset
    if not os.path.exists(DATASET_PATH):

        print("ERROR: Dataset not found.")
        print(DATASET_PATH)
        return

    # Create output directory
    os.makedirs(
        OUTPUT_PATH,
        exist_ok=True
    )

    # Supported image formats
    image_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    )

    total_images = 0
    successful_images = 0
    failed_images = 0

    print("=" * 60)
    print("DeepFishVision - Module 1")
    print("OpenCV Image Preprocessing & Enhancement")
    print("=" * 60)

    print("\nDataset:")
    print(DATASET_PATH)

    print("\nOutput:")
    print(OUTPUT_PATH)

    print("\nStarting preprocessing...\n")

    # Walk through all folders
    for root, dirs, files in os.walk(DATASET_PATH):

        for filename in files:

            if not filename.lower().endswith(
                image_extensions
            ):
                continue

            total_images += 1

            # Original image path
            input_path = os.path.join(
                root,
                filename
            )

            # Preserve folder structure
            relative_folder = os.path.relpath(
                root,
                DATASET_PATH
            )

            output_folder = os.path.join(
                OUTPUT_PATH,
                relative_folder
            )

            # Create output folder
            os.makedirs(
                output_folder,
                exist_ok=True
            )

            # Output image path
            output_path = os.path.join(
                output_folder,
                filename
            )

            try:

                # Read image
                image = cv2.imread(
                    input_path
                )

                if image is None:

                    print(
                        f"Could not read: {filename}"
                    )

                    failed_images += 1
                    continue

                # Enhance image
                enhanced_image = enhance_image(
                    image
                )

                # Save enhanced image
                success = cv2.imwrite(
                    output_path,
                    enhanced_image
                )

                if success:

                    successful_images += 1

                    print(
                        f"[{successful_images}] "
                        f"Processed: {filename}"
                    )

                else:

                    failed_images += 1

                    print(
                        f"Failed to save: {filename}"
                    )

            except Exception as error:

                failed_images += 1

                print(
                    f"Error processing "
                    f"{filename}: {error}"
                )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 60)
    print("PREPROCESSING COMPLETED")
    print("=" * 60)

    print(
        f"Total images found : {total_images}"
    )

    print(
        f"Successfully processed : "
        f"{successful_images}"
    )

    print(
        f"Failed images : {failed_images}"
    )

    print(
        f"\nEnhanced images saved to:"
    )

    print(OUTPUT_PATH)

    print("=" * 60)


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    process_dataset()