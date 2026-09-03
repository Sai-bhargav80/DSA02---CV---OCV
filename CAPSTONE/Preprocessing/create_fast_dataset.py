import os
import shutil
import random


# ============================================================
# PATHS
# ============================================================

SOURCE_DIR = r"C:\Users\veman\OneDrive\Desktop\Open CV New Slot\fish_image"

PROJECT_DIR = (
    r"C:\Users\veman\OneDrive\Desktop"
    r"\Open CV New Slot\CAPSTONE"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "fast_dataset"
)


# ============================================================
# SETTINGS
# ============================================================

TRAIN_PER_CLASS = 10
VALIDATION_PER_CLASS = 5
TEST_PER_CLASS = 5

TOTAL_REQUIRED = (
    TRAIN_PER_CLASS
    + VALIDATION_PER_CLASS
    + TEST_PER_CLASS
)

RANDOM_SEED = 42


IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# CREATE DATASET
# ============================================================

def create_fast_dataset():

    random.seed(RANDOM_SEED)

    print("=" * 70)
    print("DeepFishVision - Fast Training Dataset")
    print("=" * 70)

    print("\nSource:")
    print(SOURCE_DIR)

    print("\nOutput:")
    print(OUTPUT_DIR)


    if not os.path.exists(SOURCE_DIR):

        print("\nERROR: Source dataset not found.")

        return


    # --------------------------------------------------------
    # Create output folders
    # --------------------------------------------------------

    for split in [
        "train",
        "validation",
        "test"
    ]:

        os.makedirs(
            os.path.join(
                OUTPUT_DIR,
                split
            ),
            exist_ok=True
        )


    # --------------------------------------------------------
    # Find classes
    # --------------------------------------------------------

    class_names = []

    for item in os.listdir(SOURCE_DIR):

        item_path = os.path.join(
            SOURCE_DIR,
            item
        )

        if os.path.isdir(item_path):

            class_names.append(item)


    class_names.sort()


    print(
        f"\nClasses found: {len(class_names)}"
    )


    # ========================================================
    # PROCESS CLASSES
    # ========================================================

    included_classes = []

    skipped_classes = []


    for class_name in class_names:

        class_path = os.path.join(
            SOURCE_DIR,
            class_name
        )


        images = [

            filename

            for filename in os.listdir(
                class_path
            )

            if filename.lower().endswith(
                IMAGE_EXTENSIONS
            )

        ]


        random.shuffle(images)


        if len(images) < TOTAL_REQUIRED:

            skipped_classes.append(
                (
                    class_name,
                    len(images)
                )
            )

            continue


        included_classes.append(
            class_name
        )


        # ----------------------------------------------------
        # Select images
        # ----------------------------------------------------

        train_images = images[
            :TRAIN_PER_CLASS
        ]


        validation_images = images[
            TRAIN_PER_CLASS:
            TRAIN_PER_CLASS
            + VALIDATION_PER_CLASS
        ]


        test_images = images[
            TRAIN_PER_CLASS
            + VALIDATION_PER_CLASS:
            TOTAL_REQUIRED
        ]


        splits = {

            "train": train_images,

            "validation": validation_images,

            "test": test_images

        }


        # ----------------------------------------------------
        # Copy files
        # ----------------------------------------------------

        for split_name, selected_images in splits.items():

            destination_folder = os.path.join(

                OUTPUT_DIR,

                split_name,

                class_name

            )


            os.makedirs(
                destination_folder,
                exist_ok=True
            )


            for filename in selected_images:

                source_file = os.path.join(
                    class_path,
                    filename
                )


                destination_file = os.path.join(

                    destination_folder,

                    filename

                )


                shutil.copy2(
                    source_file,
                    destination_file
                )


        print(
            f"[ADDED] {class_name}: "
            f"10 train / 5 validation / 5 test"
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")

    print("=" * 70)
    print("FAST DATASET CREATION COMPLETED")
    print("=" * 70)

    print(
        f"Included classes: "
        f"{len(included_classes)}"
    )

    print(
        f"Skipped classes: "
        f"{len(skipped_classes)}"
    )


    if skipped_classes:

        print("\nSkipped classes:")

        for name, count in skipped_classes:

            print(
                f"  {name}: {count} images"
            )


    print(
        "\nTraining images:",
        len(included_classes)
        * TRAIN_PER_CLASS
    )


    print(
        "Validation images:",
        len(included_classes)
        * VALIDATION_PER_CLASS
    )


    print(
        "Test images:",
        len(included_classes)
        * TEST_PER_CLASS
    )


    print("\nDataset location:")

    print(OUTPUT_DIR)

    print("=" * 70)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    create_fast_dataset()