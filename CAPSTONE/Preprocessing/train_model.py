import os
import json
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from transformers import (
    ViTImageProcessor,
    ViTForImageClassification
)

from tqdm import tqdm


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_PATH = (
    r"C:\Users\veman\OneDrive\Desktop"
    r"\Open CV New Slot\CAPSTONE"
)

DATASET_PATH = os.path.join(
    PROJECT_PATH,
    "fast_dataset"
)

OUTPUT_PATH = os.path.join(
    PROJECT_PATH,
    "Outputs"
)

MODEL_PATH = os.path.join(
    OUTPUT_PATH,
    "models"
)

GRAPH_PATH = os.path.join(
    OUTPUT_PATH,
    "graphs"
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 8

EPOCHS = 3

LEARNING_RATE = 2e-5

MODEL_NAME = "google/vit-base-patch16-224"


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

os.makedirs(
    MODEL_PATH,
    exist_ok=True
)

os.makedirs(
    GRAPH_PATH,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")


print("=" * 70)
print("DeepFishVision - ViT Training")
print("=" * 70)

print("\nDevice:", device)


if device.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

else:

    print(
        "CUDA GPU not available."
    )

    print(
        "CPU training mode enabled."
    )


# ============================================================
# DATASET PATHS
# ============================================================

TRAIN_PATH = os.path.join(
    DATASET_PATH,
    "train"
)

VALIDATION_PATH = os.path.join(
    DATASET_PATH,
    "validation"
)


if not os.path.exists(TRAIN_PATH):

    print("\nERROR: Training folder not found.")

    print(TRAIN_PATH)

    print(
        "\nRun create_fast_dataset.py first."
    )

    exit()


if not os.path.exists(
    VALIDATION_PATH
):

    print(
        "\nERROR: Validation folder not found."
    )

    print(VALIDATION_PATH)

    print(
        "\nRun create_fast_dataset.py first."
    )

    exit()


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]

    )

])


validation_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]

    )

])


# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading training dataset...")

train_dataset = datasets.ImageFolder(

    TRAIN_PATH,

    transform=train_transform

)


print(
    "Loading validation dataset..."
)

validation_dataset = datasets.ImageFolder(

    VALIDATION_PATH,

    transform=validation_transform

)


# ============================================================
# CLASS INFORMATION
# ============================================================

class_names = train_dataset.classes

num_classes = len(
    class_names
)


print(
    "\nNumber of classes:",
    num_classes
)


print("\nClasses:")

for index, class_name in enumerate(
    class_names
):

    print(
        f"{index}: {class_name}"
    )


print(
    "\nTraining images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(validation_dataset)
)


# ============================================================
# VERIFY CLASS MATCHING
# ============================================================

if (
    train_dataset.classes
    != validation_dataset.classes
):

    print(
        "\nERROR: Training and validation "
        "classes do not match."
    )

    exit()


# ============================================================
# SAVE CLASS NAMES
# ============================================================

class_file = os.path.join(
    MODEL_PATH,
    "class_names.json"
)


with open(
    class_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        class_names,
        file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0

)


validation_loader = DataLoader(

    validation_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0

)


# ============================================================
# LOAD PROCESSOR
# ============================================================

print(
    "\nLoading ViT image processor..."
)


processor = ViTImageProcessor.from_pretrained(
    MODEL_NAME
)


# ============================================================
# LOAD MODEL
# ============================================================

print(
    "\nLoading pretrained Vision Transformer..."
)


model = ViTForImageClassification.from_pretrained(

    MODEL_NAME,

    num_labels=num_classes,

    ignore_mismatched_sizes=True

)


# ============================================================
# LABEL MAPPING
# ============================================================

model.config.id2label = {

    index: class_name

    for index, class_name
    in enumerate(class_names)

}


model.config.label2id = {

    class_name: index

    for index, class_name
    in enumerate(class_names)

}


model = model.to(device)


print(
    "\nVision Transformer ready."
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE

)


# ============================================================
# TRAINING HISTORY
# ============================================================

train_losses = []

validation_losses = []

train_accuracies = []

validation_accuracies = []


best_accuracy = 0.0


# ============================================================
# TRAINING LOOP
# ============================================================

print("\n")
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)


for epoch in range(EPOCHS):

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    progress_bar = tqdm(
        train_loader,
        desc="Training"
    )


    for images, labels in progress_bar:

        images = images.to(device)

        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(

            pixel_values=images,

            labels=labels

        )


        loss = outputs.loss


        loss.backward()


        optimizer.step()


        running_loss += (

            loss.item()
            * images.size(0)

        )


        predictions = torch.argmax(

            outputs.logits,

            dim=1

        )


        correct += (

            (predictions == labels)

            .sum()

            .item()

        )


        total += labels.size(0)


        progress_bar.set_postfix(

            loss=f"{loss.item():.4f}"

        )


    train_loss = (
        running_loss / total
    )


    train_accuracy = (
        correct / total
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    validation_loss = 0.0

    validation_correct = 0

    validation_total = 0


    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(device)

            labels = labels.to(device)


            outputs = model(

                pixel_values=images,

                labels=labels

            )


            loss = outputs.loss


            validation_loss += (

                loss.item()
                * images.size(0)

            )


            predictions = torch.argmax(

                outputs.logits,

                dim=1

            )


            validation_correct += (

                (predictions == labels)

                .sum()

                .item()

            )


            validation_total += labels.size(0)


    val_loss = (

        validation_loss
        / validation_total

    )


    val_accuracy = (

        validation_correct
        / validation_total

    )


    # --------------------------------------------------------
    # STORE HISTORY
    # --------------------------------------------------------

    train_losses.append(
        train_loss
    )

    validation_losses.append(
        val_loss
    )

    train_accuracies.append(
        train_accuracy
    )

    validation_accuracies.append(
        val_accuracy
    )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print(
        f"\nTraining Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Training Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Validation Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy


        model.save_pretrained(
            MODEL_PATH
        )


        processor.save_pretrained(
            MODEL_PATH
        )


        print(
            "\nBest model saved."
        )


# ============================================================
# ACCURACY GRAPH
# ============================================================

accuracy_graph = os.path.join(

    GRAPH_PATH,

    "vit_accuracy.png"

)


plt.figure(
    figsize=(10, 6)
)


plt.plot(

    range(1, EPOCHS + 1),

    train_accuracies,

    marker="o",

    label="Training Accuracy"

)


plt.plot(

    range(1, EPOCHS + 1),

    validation_accuracies,

    marker="o",

    label="Validation Accuracy"

)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Accuracy"
)


plt.title(
    "DeepFishVision - ViT Accuracy"
)


plt.legend()


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(

    accuracy_graph,

    dpi=300

)


plt.close()


# ============================================================
# LOSS GRAPH
# ============================================================

loss_graph = os.path.join(

    GRAPH_PATH,

    "vit_loss.png"

)


plt.figure(
    figsize=(10, 6)
)


plt.plot(

    range(1, EPOCHS + 1),

    train_losses,

    marker="o",

    label="Training Loss"

)


plt.plot(

    range(1, EPOCHS + 1),

    validation_losses,

    marker="o",

    label="Validation Loss"

)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
)


plt.title(
    "DeepFishVision - ViT Loss"
)


plt.legend()


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(

    loss_graph,

    dpi=300

)


plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")

print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)


print(
    f"Best Validation Accuracy: "
    f"{best_accuracy * 100:.2f}%"
)


print("\nModel:")

print(MODEL_PATH)


print("\nClass names:")

print(class_file)


print("\nAccuracy graph:")

print(accuracy_graph)


print("\nLoss graph:")

print(loss_graph)


print("=" * 70)