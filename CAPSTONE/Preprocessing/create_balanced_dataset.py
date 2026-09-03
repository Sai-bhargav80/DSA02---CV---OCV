import os
import shutil
import random


# ============================================================
# DEEPFISHVISION - BALANCED DATASET CREATION
# ============================================================

SOURCE_DIR = r"C:\Users\veman\OneDrive\Desktop\Open CV New Slot\fish_image"

PROJECT_DIR = r"C:\Users\veman\OneDrive\Desktop\Open CV New Slot\CAPSTONE"

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "balanced_dataset"
)


# ============================================================
# SETTINGS
# ============================================================

TRAIN_IMAGES_PER_CLASS = 100
VALIDATION_IMAGES_PER_CLASS = 20
TEST_IMAGES_PER_CLASS = 20

RANDOM_SEED = 42


IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# CREATE BALANCED DATASET
# ============================================================

def create_balanced_dataset():

    random.seed(RANDOM_SEED)

    print("=" * 70)
    print("DeepFishVision - Balanced Dataset Creation")
    print("=" * 70)

    print("\nSource:")
    print(SOURCE_DIR)

    print("\nOutput:")
    print(OUTPUT_DIR)


    # --------------------------------------------------------
    # Check source
    # --------------------------------------------------------

    if not os.path.exists(SOURCE_DIR):

        print("\nERROR: Dataset not found.")

        print(SOURCE_DIR)

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
    # Find species
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
        f"\nNumber of species found: "
        f"{len(class_names)}"
    )


    if len(class_names) == 0:

        print(
            "\nERROR: No species folders found."
        )

        return


    # ========================================================
    # PROCESS EACH SPECIES
    # ========================================================

    total_train = 0
    total_validation = 0
    total_test = 0


    for index, class_name in enumerate(
        class_names,
        start=1
    ):

        class_path = os.path.join(
            SOURCE_DIR,
            class_name
        )


        # Find images
        images = []

        for filename in os.listdir(
            class_path
        ):

            if filename.lower().endswith(
                IMAGE_EXTENSIONS
            ):

                images.append(filename)


        random.shuffle(images)


        total_available = len(images)


        # ----------------------------------------------------
        # Determine number of images
        # ----------------------------------------------------

        train_count = min(
            TRAIN_IMAGES_PER_CLASS,
            total_available
        )


        remaining = (
            total_available
            - train_count
        )


        validation_count = min(
            VALIDATION_IMAGES_PER_CLASS,
            remaining
        )


        remaining -= validation_count


        test_count = min(
            TEST_IMAGES_PER_CLASS,
            remaining
        )


        # ----------------------------------------------------
        # Select images
        # ----------------------------------------------------

        train_images = images[
            :train_count
        ]


        validation_images = images[
            train_count:
            train_count + validation_count
        ]


        test_images = images[
            train_count
            + validation_count:
            train_count
            + validation_count
            + test_count
        ]


        splits = {

            "train": train_images,

            "validation": validation_images,

            "test": test_images

        }


        # ----------------------------------------------------
        # Copy images
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


        total_train += train_count

        total_validation += validation_count

        total_test += test_count


        print(
            f"{index:02d}. "
            f"{class_name}: "
            f"Train={train_count}, "
            f"Validation={validation_count}, "
            f"Test={test_count}, "
            f"Available={total_available}"
        )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")

    print("=" * 70)
    print("BALANCED DATASET CREATION COMPLETED")
    print("=" * 70)

    print(
        f"Number of classes : "
        f"{len(class_names)}"
    )

    print(
        f"Training images   : "
        f"{total_train}"
    )

    print(
        f"Validation images : "
        f"{total_validation}"
    )

    print(
        f"Test images       : "
        f"{total_test}"
    )

    print("\nDataset saved at:")

    print(OUTPUT_DIR)

    print("=" * 70)


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    create_balanced_dataset()