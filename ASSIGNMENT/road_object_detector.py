"""
AUTONOMOUS ROAD SAFETY SYSTEM
PHASE 3 - ROAD OBJECT AND PEDESTRIAN DETECTION

DSA02 - Computer Vision with OpenCV

Detects:
    - Person
    - Bicycle
    - Car
    - Motorcycle
    - Bus
    - Truck

Input:
    2165-155327596_medium.mp4

Output:
    data/output/road_detection.mp4
"""

import cv2
import os
import time
from ultralytics import YOLO


# ============================================================
# 1. FILE PATHS
# ============================================================

INPUT_VIDEO = (
    r"C:\Users\veman\OneDrive\Desktop\Open CV New Slot"
    r"\Final Report\2165-155327596_medium.mp4"
)

OUTPUT_FOLDER = "data/output"

OUTPUT_VIDEO = os.path.join(
    OUTPUT_FOLDER,
    "road_detection.mp4"
)


# ============================================================
# 2. YOLO MODEL SETTINGS
# ============================================================

MODEL_PATH = "yolo11n.pt"

CONFIDENCE_THRESHOLD = 0.40


# ============================================================
# 3. ROAD SAFETY OBJECT CLASSES
# ============================================================

TARGET_CLASSES = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}


# ============================================================
# 4. CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# 5. CHECK VIDEO
# ============================================================

print("=" * 70)
print("AUTONOMOUS ROAD SAFETY SYSTEM")
print("ROAD OBJECT AND PEDESTRIAN DETECTION")
print("=" * 70)

print()
print("Input video:")
print(INPUT_VIDEO)

print()

if not os.path.exists(INPUT_VIDEO):

    print("ERROR: Input video was not found.")
    print()
    print("Please check the video path.")
    exit()


print("Input video found successfully.")


# ============================================================
# 6. LOAD YOLO MODEL
# ============================================================

print()
print("Loading YOLO model...")

try:

    model = YOLO(MODEL_PATH)

    print("YOLO model loaded successfully.")

except Exception as error:

    print()
    print("ERROR: YOLO model could not be loaded.")
    print(error)
    exit()


# ============================================================
# 7. OPEN VIDEO
# ============================================================

print()
print("Opening video...")

cap = cv2.VideoCapture(
    INPUT_VIDEO
)

if not cap.isOpened():

    print("ERROR: Could not open video.")
    exit()


# ============================================================
# 8. READ VIDEO INFORMATION
# ============================================================

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

input_fps = cap.get(
    cv2.CAP_PROP_FPS
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

video_duration = (
    total_frames / input_fps
    if input_fps > 0
    else 0
)


print()
print("=" * 70)
print("VIDEO INFORMATION")
print("=" * 70)

print(f"Width          : {width}")
print(f"Height         : {height}")
print(f"Input FPS      : {input_fps:.2f}")
print(f"Total Frames   : {total_frames}")
print(f"Duration       : {video_duration:.2f} seconds")


# ============================================================
# 9. CREATE OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

output_fps = (
    input_fps
    if input_fps > 0
    else 30
)

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    output_fps,
    (width, height)
)

if not writer.isOpened():

    print()
    print("ERROR: Could not create output video.")
    cap.release()
    exit()


# ============================================================
# 10. PERFORMANCE VARIABLES
# ============================================================

frame_count = 0

total_detections = 0

total_processing_time = 0.0

class_counts = {
    name: 0
    for name in TARGET_CLASSES.values()
}


# ============================================================
# 11. PROCESS VIDEO
# ============================================================

print()
print("=" * 70)
print("STARTING YOLO DETECTION")
print("=" * 70)

print()
print("Press Q to stop processing.")
print()


while True:

    # --------------------------------------------------------
    # READ FRAME
    # --------------------------------------------------------

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    start_time = time.perf_counter()


    # --------------------------------------------------------
    # YOLO DETECTION
    # --------------------------------------------------------

    results = model.predict(
        source=frame,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    result = results[0]


    # --------------------------------------------------------
    # FRAME DETECTION COUNT
    # --------------------------------------------------------

    frame_detections = 0


    # --------------------------------------------------------
    # PROCESS DETECTIONS
    # --------------------------------------------------------

    if result.boxes is not None:

        for box in result.boxes:

            # Class ID

            class_id = int(
                box.cls[0]
            )


            # Confidence

            confidence = float(
                box.conf[0]
            )


            # Only road-safety classes

            if class_id not in TARGET_CLASSES:

                continue


            label = TARGET_CLASSES[class_id]


            # Bounding box coordinates

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # ------------------------------------------------
            # DRAW BOUNDING BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # ------------------------------------------------
            # CREATE LABEL
            # ------------------------------------------------

            label_text = (
                f"{label}: "
                f"{confidence:.2f}"
            )


            # ------------------------------------------------
            # LABEL BACKGROUND
            # ------------------------------------------------

            (text_width, text_height), baseline = (
                cv2.getTextSize(
                    label_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    2
                )
            )


            label_y = max(
                y1,
                text_height + 10
            )


            cv2.rectangle(
                frame,
                (
                    x1,
                    label_y - text_height - 10
                ),
                (
                    x1 + text_width + 10,
                    label_y
                ),
                (0, 255, 0),
                -1
            )


            # ------------------------------------------------
            # DRAW LABEL
            # ------------------------------------------------

            cv2.putText(
                frame,
                label_text,
                (
                    x1 + 5,
                    label_y - 5
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2
            )


            # ------------------------------------------------
            # UPDATE COUNTS
            # ------------------------------------------------

            frame_detections += 1

            total_detections += 1

            class_counts[label] += 1


    # ========================================================
    # 12. CALCULATE PROCESSING TIME
    # ========================================================

    elapsed_time = (
        time.perf_counter()
        - start_time
    )

    total_processing_time += elapsed_time


    processing_fps = (
        1.0 / elapsed_time
        if elapsed_time > 0
        else 0
    )


    # ========================================================
    # 13. INFORMATION PANEL
    # ========================================================

    panel_height = 135

    cv2.rectangle(
        frame,
        (10, 10),
        (400, panel_height),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        "ROAD SAFETY VISION",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Frame: {frame_count}/{total_frames}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Objects: {frame_detections}",
        (20, 83),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"FPS: {processing_fps:.2f}",
        (20, 106),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Confidence: {CONFIDENCE_THRESHOLD}",
        (20, 129),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # ========================================================
    # 14. SAVE OUTPUT FRAME
    # ========================================================

    writer.write(
        frame
    )


    # ========================================================
    # 15. DISPLAY FRAME
    # ========================================================

    cv2.imshow(
        "Autonomous Road Safety System",
        frame
    )


    # ========================================================
    # 16. KEY CONTROL
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        print()
        print("Processing stopped by user.")
        break


# ============================================================
# 17. RELEASE RESOURCES
# ============================================================

cap.release()

writer.release()

cv2.destroyAllWindows()


# ============================================================
# 18. FINAL PERFORMANCE CALCULATION
# ============================================================

if frame_count > 0:

    average_fps = (
        frame_count
        / total_processing_time
    )

    average_latency = (
        total_processing_time
        / frame_count
    )

else:

    average_fps = 0

    average_latency = 0


# ============================================================
# 19. DISPLAY RESULTS
# ============================================================

print()
print("=" * 70)
print("ROAD OBJECT DETECTION COMPLETED")
print("=" * 70)

print()

print("PROCESSING RESULTS")
print("-" * 70)

print(
    f"Frames processed       : {frame_count}"
)

print(
    f"Total detections       : {total_detections}"
)

print(
    f"Average processing FPS : {average_fps:.2f}"
)

print(
    f"Average latency        : "
    f"{average_latency * 1000:.2f} ms"
)


print()
print("OBJECT DETECTION COUNTS")
print("-" * 70)

for class_name, count in class_counts.items():

    print(
        f"{class_name:15s}: {count}"
    )


print()
print("OUTPUT FILE")
print("-" * 70)

print(
    os.path.abspath(OUTPUT_VIDEO)
)


print()
print("=" * 70)
print("PHASE 3 - OBJECT & PEDESTRIAN DETECTION: PASSED")
print("=" * 70)
