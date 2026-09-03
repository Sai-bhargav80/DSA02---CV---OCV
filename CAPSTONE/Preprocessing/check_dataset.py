import os


DATASET_PATH = r"C:\Users\veman\OneDrive\Desktop\Open CV New Slot\fish_image"


print("=" * 60)
print("DeepFishVision - Dataset Structure Checker")
print("=" * 60)


if not os.path.exists(DATASET_PATH):
    print("\nERROR: Dataset path does not exist.")
    print(DATASET_PATH)
    exit()


print("\nDataset found successfully!")
print("Dataset path:")
print(DATASET_PATH)


print("\nFolders and files:")
print("-" * 60)


for root, dirs, files in os.walk(DATASET_PATH):

    level = root.replace(
        DATASET_PATH, ""
    ).count(os.sep)

    indent = "    " * level

    folder_name = os.path.basename(root)

    print(f"{indent}[Folder] {folder_name}")

    sub_indent = "    " * (level + 1)

    for file in files[:5]:

        print(
            f"{sub_indent}{file}"
        )

    if len(files) > 5:

        print(
            f"{sub_indent}... "
            f"{len(files) - 5} more files"
        )


print("\n" + "=" * 60)
print("Dataset structure check completed.")
print("=" * 60)