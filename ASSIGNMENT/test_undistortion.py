import cv2
import numpy as np
import os


# =========================================================
# CONFIGURATION
# =========================================================

CALIBRATION_FILE = "calibration/results/camera_calibration.npz"
OUTPUT_FOLDER = "calibration/results"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# =========================================================
# CHECK CALIBRATION FILE
# =========================================================

if not os.path.exists(CALIBRATION_FILE):

    print("=" * 60)
    print("ERROR: Calibration file not found.")
    print("=" * 60)

    print("Expected:")
    print(CALIBRATION_FILE)

    print()
    print("Run calibrate_camera.py first.")

    exit()


# =========================================================
# LOAD CALIBRATION PARAMETERS
# =========================================================

data = np.load(CALIBRATION_FILE)

camera_matrix = data["camera_matrix"]
distortion_coefficients = data["distortion_coefficients"]


print("=" * 60)
print("CAMERA UNDISTORTION TEST")
print("=" * 60)

print("Calibration parameters loaded successfully.")


# =========================================================
# OPEN CAMERA
# =========================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Camera could not be opened.")
    exit()


print()
print("Camera opened successfully.")
print()
print("Controls:")
print("Q = Quit")
print("S = Save comparison image")
print("=" * 60)


# =========================================================
# CAMERA LOOP
# =========================================================

while True:

    ret, frame = camera.read()

    if not ret:

        print("ERROR: Could not read camera frame.")
        break


    # -----------------------------------------------------
    # GET FRAME SIZE
    # -----------------------------------------------------

    height, width = frame.shape[:2]


    # -----------------------------------------------------
    # CALCULATE OPTIMAL CAMERA MATRIX
    # -----------------------------------------------------

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        distortion_coefficients,
        (width, height),
        1,
        (width, height)
    )


    # -----------------------------------------------------
    # UNDISTORT IMAGE
    # -----------------------------------------------------

    undistorted = cv2.undistort(
        frame,
        camera_matrix,
        distortion_coefficients,
        None,
        new_camera_matrix
    )


    # -----------------------------------------------------
    # CREATE SIDE-BY-SIDE COMPARISON
    # -----------------------------------------------------

    comparison = np.hstack(
        (frame, undistorted)
    )


    # -----------------------------------------------------
    # ADD LABELS
    # -----------------------------------------------------

    cv2.putText(
        comparison,
        "ORIGINAL",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        comparison,
        "UNDISTORTED",
        (width + 30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    # -----------------------------------------------------
    # DISPLAY
    # -----------------------------------------------------

    cv2.imshow(
        "Camera Calibration - Original vs Undistorted",
        comparison
    )


    key = cv2.waitKey(1) & 0xFF


    # -----------------------------------------------------
    # SAVE RESULT
    # -----------------------------------------------------

    if key == ord("s"):

        output_path = os.path.join(
            OUTPUT_FOLDER,
            "undistortion_comparison.jpg"
        )

        cv2.imwrite(
            output_path,
            comparison
        )

        print()
        print("Comparison image saved:")
        print(output_path)


    # -----------------------------------------------------
    # QUIT
    # -----------------------------------------------------

    elif key == ord("q"):

        break


# =========================================================
# CLEANUP
# =========================================================

camera.release()
cv2.destroyAllWindows()

print()
print("=" * 60)
print("UNDISTORTION TEST COMPLETED")
print("=" * 60)
