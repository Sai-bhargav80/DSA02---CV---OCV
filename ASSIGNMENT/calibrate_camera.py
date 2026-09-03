import cv2
import numpy as np
import glob
import os


# =========================================================
# CAMERA CALIBRATION CONFIGURATION
# =========================================================

IMAGE_FOLDER = "calibration/images"
RESULT_FOLDER = "calibration/results"

# IMPORTANT:
# Change this if your chessboard has a different number
# of INTERNAL corners.
CHESSBOARD_SIZE = (9, 6)

# Size of one chessboard square.
# The actual physical size is not important for basic
# calibration as long as it is consistent.
SQUARE_SIZE = 1.0


# =========================================================
# CREATE RESULT FOLDER
# =========================================================

os.makedirs(RESULT_FOLDER, exist_ok=True)


# =========================================================
# PREPARE 3D OBJECT POINTS
# =========================================================

object_points = np.zeros(
    (CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3),
    np.float32
)

object_points[:, :2] = np.mgrid[
    0:CHESSBOARD_SIZE[0],
    0:CHESSBOARD_SIZE[1]
].T.reshape(-1, 2)

object_points *= SQUARE_SIZE


# =========================================================
# STORAGE FOR CALIBRATION POINTS
# =========================================================

object_points_list = []
image_points_list = []


# =========================================================
# FIND CALIBRATION IMAGES
# =========================================================

image_paths = glob.glob(
    os.path.join(IMAGE_FOLDER, "*.jpg")
)

image_paths += glob.glob(
    os.path.join(IMAGE_FOLDER, "*.png")
)

print("=" * 65)
print("CAMERA CALIBRATION")
print("=" * 65)

print(f"Images found: {len(image_paths)}")

if len(image_paths) == 0:

    print()
    print("ERROR: No calibration images found.")
    print()
    print("Place calibration images inside:")
    print("calibration/images/")
    print()
    exit()


# =========================================================
# PROCESS EACH IMAGE
# =========================================================

successful_images = 0

image_size = None

for image_path in image_paths:

    print()
    print(f"Processing: {image_path}")

    image = cv2.imread(image_path)

    if image is None:

        print("Could not read image.")
        continue

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    image_size = gray.shape[::-1]

    # -----------------------------------------------------
    # FIND CHESSBOARD CORNERS
    # -----------------------------------------------------

    found, corners = cv2.findChessboardCorners(
        gray,
        CHESSBOARD_SIZE,
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK
    )

    if found:

        # -------------------------------------------------
        # REFINE CORNER LOCATIONS
        # -------------------------------------------------

        criteria = (
            cv2.TERM_CRITERIA_EPS
            + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )

        refined_corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        object_points_list.append(
            object_points.copy()
        )

        image_points_list.append(
            refined_corners
        )

        successful_images += 1

        print("Chessboard detected: YES")

        # -------------------------------------------------
        # DRAW DETECTED CORNERS
        # -------------------------------------------------

        visualization = image.copy()

        cv2.drawChessboardCorners(
            visualization,
            CHESSBOARD_SIZE,
            refined_corners,
            found
        )

        filename = os.path.basename(image_path)

        output_path = os.path.join(
            RESULT_FOLDER,
            "corners_" + filename
        )

        cv2.imwrite(
            output_path,
            visualization
        )

    else:

        print("Chessboard detected: NO")


# =========================================================
# CHECK SUCCESSFUL DETECTIONS
# =========================================================

print()
print("=" * 65)
print("CALIBRATION IMAGE ANALYSIS")
print("=" * 65)

print(f"Total images       : {len(image_paths)}")
print(f"Successful images  : {successful_images}")

if successful_images < 5:

    print()
    print("ERROR: Not enough valid calibration images.")
    print()
    print("Try:")
    print("1. Checking the chessboard size.")
    print("2. Capturing clearer images.")
    print("3. Ensuring the complete chessboard is visible.")
    print("4. Using different angles.")
    print()
    exit()


# =========================================================
# PERFORM CAMERA CALIBRATION
# =========================================================

print()
print("Calculating camera parameters...")

ret, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = cv2.calibrateCamera(
    object_points_list,
    image_points_list,
    image_size,
    None,
    None
)


# =========================================================
# PRINT RESULTS
# =========================================================

print()
print("=" * 65)
print("CAMERA CALIBRATION RESULTS")
print("=" * 65)

print()
print("Camera Matrix:")
print(camera_matrix)

print()
print("Distortion Coefficients:")
print(distortion_coefficients)

print()
print(f"Calibration RMS Error: {ret}")


# =========================================================
# CALCULATE RE-PROJECTION ERROR
# =========================================================

total_error = 0

for i in range(len(object_points_list)):

    projected_points, _ = cv2.projectPoints(
        object_points_list[i],
        rotation_vectors[i],
        translation_vectors[i],
        camera_matrix,
        distortion_coefficients
    )

    error = cv2.norm(
        image_points_list[i],
        projected_points,
        cv2.NORM_L2
    ) / len(projected_points)

    total_error += error


mean_error = total_error / len(object_points_list)


print()
print(f"Mean Re-projection Error: {mean_error:.6f}")


# =========================================================
# SAVE CALIBRATION PARAMETERS
# =========================================================

output_file = os.path.join(
    RESULT_FOLDER,
    "camera_calibration.npz"
)

np.savez(
    output_file,
    camera_matrix=camera_matrix,
    distortion_coefficients=distortion_coefficients,
    image_width=image_size[0],
    image_height=image_size[1],
    reprojection_error=mean_error
)


# =========================================================
# SAVE TEXT REPORT
# =========================================================

report_file = os.path.join(
    RESULT_FOLDER,
    "calibration_report.txt"
)

with open(report_file, "w") as file:

    file.write("CAMERA CALIBRATION REPORT\n")
    file.write("=" * 60 + "\n\n")

    file.write(
        f"Calibration Images: {len(image_paths)}\n"
    )

    file.write(
        f"Successful Images: {successful_images}\n\n"
    )

    file.write("Camera Matrix:\n")
    file.write(
        np.array2string(camera_matrix)
    )

    file.write("\n\nDistortion Coefficients:\n")
    file.write(
        np.array2string(distortion_coefficients)
    )

    file.write("\n\n")
    file.write(
        f"RMS Error: {ret}\n"
    )

    file.write(
        f"Mean Re-projection Error: {mean_error}\n"
    )


# =========================================================
# FINAL MESSAGE
# =========================================================

print()
print("=" * 65)
print("CAMERA CALIBRATION COMPLETED SUCCESSFULLY")
print("=" * 65)

print()
print("Saved files:")

print(
    f"1. {output_file}"
)

print(
    f"2. {report_file}"
)

print(
    f"3. {RESULT_FOLDER}/corners_*.jpg"
)

print()
print("PHASE 2 - CAMERA CALIBRATION: PASSED")
print("=" * 65)
