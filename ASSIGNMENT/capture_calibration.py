import cv2
import os
import time

OUTPUT_FOLDER = "calibration/images"
CAMERA_INDEX = 0

TARGET_IMAGES = 20

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("=" * 60)
print("CAMERA CALIBRATION IMAGE CAPTURE")
print("=" * 60)

print("Press SPACE to capture an image")
print("Press Q to quit")
print(f"Target images: {TARGET_IMAGES}")
print("=" * 60)

count = 0

while True:

    ret, frame = camera.read()

    if not ret:
        print("ERROR: Cannot read camera.")
        break

    display = frame.copy()

    cv2.putText(
        display,
        f"Captured: {count}/{TARGET_IMAGES}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        "SPACE = Capture | Q = Quit",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("Camera Calibration", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):

        count += 1

        filename = os.path.join(
            OUTPUT_FOLDER,
            f"calibration_{count:02d}.jpg"
        )

        cv2.imwrite(filename, frame)

        print(f"Saved: {filename}")

        time.sleep(0.3)

        if count >= TARGET_IMAGES:
            print("20 images captured successfully.")
            break

    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("=" * 60)
print(f"Total images captured: {count}")
print(f"Saved in: {OUTPUT_FOLDER}")
print("=" * 60)
