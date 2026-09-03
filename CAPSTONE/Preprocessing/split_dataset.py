import os
import shutil
import random


# ============================================================
# DEEPFISHVISION - DATASET SPLITTING
# ============================================================

SOURCE_DIR = r"C:\Users\veman\OneDrive\Desktop\Open CV New Slot\fish_image"

PROJECT_DIR = r"C:\Users\veman\OneDrive\Desktop\Open CV New Slot\CAPSTONE"

DESTINATION_DIR = os.path.join(
    PROJECT_DIR,
    "dataset"
)


# ============================================================
# SPLIT RATIOS
# ============================================================

TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10

RANDOM_SEED = 42


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# DATASET SPLITTING FUNCTION
# ============================================================

def split_dataset():

    random.seed(RANDOM_SEED)

    # Check source dataset
    if not os.path.exists(SOURCE_DIR):

        print("ERROR: Dataset not found.")
        print(SOURCE_DIR)
        return

    print("=" * 70)
    print("DeepFishVision - Dataset Splitting")
    print("=" * 70)

    print("\nSource:")
    print(SOURCE_DIR)

    print("\nDestination:")
    print(DESTINATION_DIR)


    # --------------------------------------------------------
    # Create destination folders
    # --------------------------------------------------------

    for split in [
        "train",
        "validation",
        "test"
    ]:

        os.makedirs(
            os.path.join(
                DESTINATION_DIR,
                split
            ),
            exist_ok=True
        )


    # --------------------------------------------------------
    # Find species folders
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


    if len(class_names) == 0:

        print(
            "\nERROR: No species folders found."
        )

        return


    print(
        f"\nNumber of species: {len(class_names)}"
    )

    print("\nSpecies:")

    for index, class_name in enumerate(
        class_names,
        start=1
    ):

        print(
            f"{index}. {class_name}"
        )


    total_images = 0


    # ========================================================
    # PROCESS EACH SPECIES
    # ========================================================

    for class_name in class_names:

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


        # Randomize
        random.shuffle(images)


        total_class_images = len(images)


        # Calculate split sizes
        train_count = int(
            total_class_images
            * TRAIN_RATIO
        )

        validation_count = int(
            total_class_images
            * VALIDATION_RATIO
        )


        # Create splits
        train_images = images[
            :train_count
        ]

        validation_images = images[
            train_count:
            train_count + validation_count
        ]

        test_images = images[
            train_count + validation_count:
        ]


        splits = {

            "train": train_images,

            "validation": validation_images,

            "test": test_images

        }


        print("\n" + "-" * 70)

        print(
            f"{class_name}: "
            f"{total_class_images} images"
        )


        # ----------------------------------------------------
        # Copy images
        # ----------------------------------------------------

        for split_name, split_images in splits.items():

            destination_class = os.path.join(

                DESTINATION_DIR,

                split_name,

                class_name

            )


            os.makedirs(
                destination_class,
                exist_ok=True
            )


            for filename in split_images:

                source_file = os.path.join(
                    class_path,
                    filename
                )


                destination_file = os.path.join(

                    destination_class,

                    filename

                )


                shutil.copy2(
                    source_file,
                    destination_file
                )


            print(
                f"  {split_name:<12}: "
                f"{len(split_images)}"
            )


        total_images += total_class_images


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")

    print("=" * 70)
    print("DATASET SPLITTING COMPLETED")
    print("=" * 70)

    print(
        f"Total images : {total_images}"
    )

    print(
        f"Total classes: {len(class_names)}"
    )

    print("\nTrain:")
    print(
        os.path.join(
            DESTINATION_DIR,
            "train"
        )
    )

    print("\nValidation:")
    print(
        os.path.join(
            DESTINATION_DIR,
            "validation"
        )
    )

    print("\nTest:")
    print(
        os.path.join(
            DESTINATION_DIR,
            "test"
        )
    )

    print("\n" + "=" * 70)


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    split_dataset()