import os
import json
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_PATH = (
    r"C:\Users\veman\OneDrive\Desktop"
    r"\Open CV New Slot\CAPSTONE"
)

MODEL_PATH = os.path.join(
    PROJECT_PATH,
    "Outputs",
    "models"
)

PREDICTION_OUTPUT = os.path.join(
    PROJECT_PATH,
    "Outputs",
    "predictions"
)


# ============================================================
# TEST IMAGE
# ============================================================
# CHANGE THIS TO THE IMAGE YOU WANT TO TEST

IMAGE_PATH = (
    r"C:\Users\veman\OneDrive\Desktop"
    r"\Open CV New Slot\fish_image"
)


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    PREDICTION_OUTPUT,
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
print("DeepFishVision - Fish Species Prediction")
print("=" * 70)

print(
    "\nDevice:",
    device
)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(
    MODEL_PATH
):

    print(
        "\nERROR: Trained model not found."
    )

    print(MODEL_PATH)

    exit()


# ============================================================
# LOAD CLASS NAMES
# ============================================================

class_file = os.path.join(
    MODEL_PATH,
    "class_names.json"
)


with open(
    class_file,
    "r",
    encoding="utf-8"
) as file:

    class_names = json.load(file)


print(
    "\nNumber of classes:",
    len(class_names)
)


# ============================================================
# LOAD PROCESSOR
# ============================================================

print(
    "\nLoading image processor..."
)


processor = ViTImageProcessor.from_pretrained(
    MODEL_PATH
)


# ============================================================
# LOAD MODEL
# ============================================================

print(
    "Loading trained ViT model..."
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
# CHECK IMAGE
# ============================================================

if not os.path.isfile(
    IMAGE_PATH
):

    print(
        "\nERROR: Test image not found."
    )

    print(
        "\nCurrent image path:"
    )

    print(IMAGE_PATH)

    print(
        "\nChange IMAGE_PATH in "
        "predict_fish.py to an actual image."
    )

    exit()


# ============================================================
# READ IMAGE USING OPENCV
# ============================================================

image = cv2.imread(
    IMAGE_PATH
)


if image is None:

    print(
        "\nERROR: OpenCV could not read image."
    )

    exit()


# ============================================================
# OPENCV PREPROCESSING
# ============================================================

# Resize
resized = cv2.resize(
    image,
    (224, 224)
)


# Noise reduction
denoised = cv2.GaussianBlur(
    resized,
    (3, 3),
    0
)


# Convert BGR → LAB
lab = cv2.cvtColor(
    denoised,
    cv2.COLOR_BGR2LAB
)


# Split channels
l, a, b = cv2.split(
    lab
)


# CLAHE
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)


enhanced_l = clahe.apply(
    l
)


# Merge
enhanced_lab = cv2.merge(
    (enhanced_l, a, b)
)


# LAB → RGB
enhanced_rgb = cv2.cvtColor(
    enhanced_lab,
    cv2.COLOR_LAB2RGB
)


# ============================================================
# CONVERT IMAGE FOR ViT
# ============================================================

pil_image = Image.fromarray(
    enhanced_rgb
)


inputs = processor(
    images=pil_image,
    return_tensors="pt"
)


pixel_values = inputs[
    "pixel_values"
].to(device)


# ============================================================
# PREDICTION
# ============================================================

print(
    "\nRunning prediction..."
)


with torch.no_grad():

    outputs = model(
        pixel_values=pixel_values
    )


# ============================================================
# PROBABILITIES
# ============================================================

probabilities = torch.nn.functional.softmax(
    outputs.logits,
    dim=1
)


# Top 5 predictions

top_probabilities, top_indices = torch.topk(
    probabilities,
    k=min(5, len(class_names)),
    dim=1
)


top_probabilities = (
    top_probabilities[0]
    .cpu()
    .numpy()
)


top_indices = (
    top_indices[0]
    .cpu()
    .numpy()
)


# Best prediction

best_index = top_indices[0]

best_species = class_names[
    best_index
]

best_confidence = (
    top_probabilities[0]
    * 100
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n")

print("=" * 70)
print("PREDICTION RESULT")
print("=" * 70)


print(
    "\nPredicted Species:"
)


print(
    best_species
)


print(
    f"\nConfidence: "
    f"{best_confidence:.2f}%"
)


print("\nTop 5 Predictions:")

print("-" * 50)


for rank, (index, probability) in enumerate(
    zip(
        top_indices,
        top_probabilities
    ),
    start=1
):

    print(
        f"{rank}. "
        f"{class_names[index]} "
        f"- "
        f"{probability * 100:.2f}%"
    )


# ============================================================
# VISUALIZATION
# ============================================================

plt.figure(
    figsize=(12, 6)
)


# ------------------------------------------------------------
# Enhanced image
# ------------------------------------------------------------

plt.subplot(
    1,
    2,
    1
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


# ------------------------------------------------------------
# Prediction chart
# ------------------------------------------------------------

plt.subplot(
    1,
    2,
    2
)


top_names = [

    class_names[index]

    for index in top_indices

]


top_percentages = (

    top_probabilities * 100

)


plt.barh(
    top_names[::-1],
    top_percentages[::-1]
)


plt.xlabel(
    "Confidence (%)"
)


plt.title(
    "Top 5 Species Predictions"
)


plt.tight_layout()


# ============================================================
# SAVE RESULT
# ============================================================

output_file = os.path.join(
    PREDICTION_OUTPUT,
    "fish_prediction.png"
)


plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n")

print("=" * 70)

print(
    "Prediction visualization saved!"
)

print("=" * 70)

print(
    output_file
)

print("=" * 70)