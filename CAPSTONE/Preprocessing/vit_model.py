import torch
from transformers import ViTForImageClassification


# ============================================================
# VISION TRANSFORMER MODEL
# ============================================================

def create_vit_model(num_classes, class_names):

    print("=" * 60)
    print("DeepFishVision - Vision Transformer")
    print("=" * 60)

    print("\nNumber of classes:", num_classes)
    print("Classes:")

    for index, class_name in enumerate(class_names):
        print(f"{index}: {class_name}")

    print("\nLoading pretrained Vision Transformer...")

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )

    # Store class information
    model.config.id2label = {
        index: class_name
        for index, class_name in enumerate(class_names)
    }

    model.config.label2id = {
        class_name: index
        for index, class_name in enumerate(class_names)
    }

    print("\nVision Transformer loaded successfully.")

    return model


# ============================================================
# TEST MODEL
# ============================================================

if __name__ == "__main__":

    example_classes = [
        "Species_1",
        "Species_2",
        "Species_3"
    ]

    model = create_vit_model(
        len(example_classes),
        example_classes
    )

    print("\nModel ready for training.")