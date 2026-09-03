import os
import json

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from transformers import (
    ViTForImageClassification,
    ViTImageProcessor
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


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

MODEL_PATH = os.path.join(
    PROJECT_PATH,
    "Outputs",
    "models"
)

OUTPUT_PATH = os.path.join(
    PROJECT_PATH,
    "Outputs"
)

GRAPH_PATH = os.path.join(
    OUTPUT_PATH,
    "graphs"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 8


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    GRAPH_PATH,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("DeepFishVision - Model Evaluation")
print("=" * 70)

print(
    "\nDevice:",
    device
)


# ============================================================
# TEST DATASET
# ============================================================

TEST_PATH = os.path.join(
    DATASET_PATH,
    "test"
)


if not os.path.exists(TEST_PATH):

    print(
        "\nERROR: Test dataset not found."
    )

    print(TEST_PATH)

    exit()


# ============================================================
# TRANSFORMATION
# ============================================================

test_transform = transforms.Compose([

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
# LOAD TEST DATASET
# ============================================================

print(
    "\nLoading test dataset..."
)


test_dataset = datasets.ImageFolder(

    TEST_PATH,

    transform=test_transform

)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0

)


class_names = test_dataset.classes


print(
    "\nNumber of classes:",
    len(class_names)
)


print(
    "Test images:",
    len(test_dataset)
)


# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(
    MODEL_PATH
):

    print(
        "\nERROR: Model not found."
    )

    print(MODEL_PATH)

    exit()


print(
    "\nLoading trained ViT model..."
)


model = ViTForImageClassification.from_pretrained(
    MODEL_PATH
)


model = model.to(device)

model.eval()


print(
    "Model loaded successfully."
)


# ============================================================
# PREDICTION
# ============================================================

all_predictions = []

all_labels = []


print(
    "\nRunning predictions..."
)


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        labels = labels.to(device)


        outputs = model(
            pixel_values=images
        )


        predictions = torch.argmax(
            outputs.logits,
            dim=1
        )


        all_predictions.extend(
            predictions.cpu().numpy()
        )


        all_labels.extend(
            labels.cpu().numpy()
        )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)


precision = precision_score(

    all_labels,

    all_predictions,

    average="weighted",

    zero_division=0

)


recall = recall_score(

    all_labels,

    all_predictions,

    average="weighted",

    zero_division=0

)


f1 = f1_score(

    all_labels,

    all_predictions,

    average="weighted",

    zero_division=0

)


# ============================================================
# DISPLAY METRICS
# ============================================================

print("\n")

print("=" * 70)
print("EVALUATION RESULTS")
print("=" * 70)


print(
    f"\nAccuracy : "
    f"{accuracy * 100:.2f}%"
)


print(
    f"Precision: "
    f"{precision * 100:.2f}%"
)


print(
    f"Recall   : "
    f"{recall * 100:.2f}%"
)


print(
    f"F1 Score : "
    f"{f1 * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")

print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)


report = classification_report(

    all_labels,

    all_predictions,

    target_names=class_names,

    zero_division=0

)


print(report)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_path = os.path.join(

    OUTPUT_PATH,

    "classification_report.txt"

)


with open(

    report_path,

    "w",

    encoding="utf-8"

) as file:

    file.write(
        "DeepFishVision - Classification Report\n"
    )

    file.write(
        "=" * 60
        + "\n\n"
    )

    file.write(
        f"Accuracy : {accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Precision: {precision * 100:.2f}%\n"
    )

    file.write(
        f"Recall   : {recall * 100:.2f}%\n"
    )

    file.write(
        f"F1 Score : {f1 * 100:.2f}%\n\n"
    )

    file.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print(
    "\nGenerating confusion matrix..."
)


cm = confusion_matrix(

    all_labels,

    all_predictions

)


# Large figure for 22 classes
plt.figure(
    figsize=(18, 15)
)


display = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=class_names

)


display.plot(

    xticks_rotation=90,

    values_format="d"

)


plt.title(
    "DeepFishVision - Confusion Matrix"
)


plt.tight_layout()


confusion_path = os.path.join(

    GRAPH_PATH,

    "confusion_matrix.png"

)


plt.savefig(

    confusion_path,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# METRICS GRAPH
# ============================================================

metric_names = [

    "Accuracy",

    "Precision",

    "Recall",

    "F1 Score"

]


metric_values = [

    accuracy * 100,

    precision * 100,

    recall * 100,

    f1 * 100

]


plt.figure(
    figsize=(10, 6)
)


plt.bar(

    metric_names,

    metric_values

)


plt.ylabel(
    "Score (%)"
)


plt.ylim(
    0,
    100
)


plt.title(
    "DeepFishVision - Model Performance"
)


for index, value in enumerate(
    metric_values
):

    plt.text(

        index,

        value + 1,

        f"{value:.2f}%",

        ha="center"

    )


plt.tight_layout()


metrics_path = os.path.join(

    GRAPH_PATH,

    "model_metrics.png"

)


plt.savefig(

    metrics_path,

    dpi=300

)


plt.close()


# ============================================================
# SAVE METRICS JSON
# ============================================================

metrics_data = {

    "accuracy": accuracy,

    "precision": precision,

    "recall": recall,

    "f1_score": f1,

    "test_images": len(test_dataset),

    "classes": len(class_names)

}


metrics_json = os.path.join(

    OUTPUT_PATH,

    "metrics.json"

)


with open(

    metrics_json,

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        metrics_data,

        file,

        indent=4

    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")

print("=" * 70)

print("MODEL EVALUATION COMPLETED")

print("=" * 70)


print(
    f"\nAccuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall   : {recall * 100:.2f}%"
)

print(
    f"F1 Score : {f1 * 100:.2f}%"
)


print("\nSaved files:")


print(
    "\nClassification report:"
)

print(report_path)


print(
    "\nConfusion matrix:"
)

print(confusion_path)


print(
    "\nPerformance metrics:"
)

print(metrics_path)


print(
    "\nMetrics JSON:"
)

print(metrics_json)


print("\n" + "=" * 70)