"""
PHASE 4 - TRAFFIC SIGN RECOGNITION

Autonomous Road Safety System
DSA02 - Computer Vision with OpenCV

Detects possible traffic signs using:
    1. HSV colour segmentation
    2. Contour detection
    3. Shape analysis
    4. Circular/triangular geometry

This module is designed to be integrated with
the main road-safety video pipeline.
"""

import cv2
import numpy as np


# ============================================================
# TRAFFIC SIGN DETECTOR
# ============================================================

class TrafficSignDetector:

    def __init__(self):

        # Minimum contour area to consider
        self.min_area = 300

        # Maximum contour area
        self.max_area = 50000


    # ========================================================
    # RED MASK
    # ========================================================

    def create_red_mask(self, hsv):

        # Lower red range
        lower_red_1 = np.array(
            [0, 80, 60]
        )

        upper_red_1 = np.array(
            [10, 255, 255]
        )

        # Upper red range
        lower_red_2 = np.array(
            [170, 80, 60]
        )

        upper_red_2 = np.array(
            [180, 255, 255]
        )

        mask1 = cv2.inRange(
            hsv,
            lower_red_1,
            upper_red_1
        )

        mask2 = cv2.inRange(
            hsv,
            lower_red_2,
            upper_red_2
        )

        mask = cv2.bitwise_or(
            mask1,
            mask2
        )

        return mask


    # ========================================================
    # BLUE MASK
    # ========================================================

    def create_blue_mask(self, hsv):

        lower_blue = np.array(
            [90, 70, 50]
        )

        upper_blue = np.array(
            [135, 255, 255]
        )

        mask = cv2.inRange(
            hsv,
            lower_blue,
            upper_blue
        )

        return mask


    # ========================================================
    # CLEAN MASK
    # ========================================================

    def clean_mask(self, mask):

        kernel = np.ones(
            (5, 5),
            np.uint8
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        return mask


    # ========================================================
    # SHAPE CLASSIFICATION
    # ========================================================

    def classify_shape(self, contour):

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter == 0:

            return "Unknown"


        approximation = cv2.approxPolyDP(
            contour,
            0.04 * perimeter,
            True
        )

        vertices = len(
            approximation
        )


        # ----------------------------------------------------
        # TRIANGLE
        # ----------------------------------------------------

        if vertices == 3:

            return "Triangle"


        # ----------------------------------------------------
        # CIRCLE
        # ----------------------------------------------------

        area = cv2.contourArea(
            contour
        )

        circularity = (
            4 * np.pi * area
            / (perimeter * perimeter)
        )

        if circularity > 0.70:

            return "Circle"


        # ----------------------------------------------------
        # RECTANGLE / SQUARE
        # ----------------------------------------------------

        if vertices == 4:

            return "Rectangle"


        return "Unknown"


    # ========================================================
    # DETECT SIGNS
    # ========================================================

    def detect(self, frame):

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )


        # ----------------------------------------------------
        # CREATE COLOUR MASKS
        # ----------------------------------------------------

        red_mask = self.create_red_mask(
            hsv
        )

        blue_mask = self.create_blue_mask(
            hsv
        )


        red_mask = self.clean_mask(
            red_mask
        )

        blue_mask = self.clean_mask(
            blue_mask
        )


        # ----------------------------------------------------
        # COMBINE MASKS
        # ----------------------------------------------------

        combined_mask = cv2.bitwise_or(
            red_mask,
            blue_mask
        )


        # ----------------------------------------------------
        # FIND CONTOURS
        # ----------------------------------------------------

        contours, _ = cv2.findContours(
            combined_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )


        detections = []


        # ====================================================
        # PROCESS CONTOURS
        # ====================================================

        for contour in contours:

            area = cv2.contourArea(
                contour
            )


            if area < self.min_area:

                continue


            if area > self.max_area:

                continue


            # ------------------------------------------------
            # BOUNDING RECTANGLE
            # ------------------------------------------------

            x, y, w, h = cv2.boundingRect(
                contour
            )


            if w == 0 or h == 0:

                continue


            # ------------------------------------------------
            # ASPECT RATIO
            # ------------------------------------------------

            aspect_ratio = w / float(h)


            if aspect_ratio < 0.5:

                continue


            if aspect_ratio > 1.5:

                continue


            # ------------------------------------------------
            # SHAPE
            # ------------------------------------------------

            shape = self.classify_shape(
                contour
            )


            if shape == "Unknown":

                continue


            # ------------------------------------------------
            # DETERMINE COLOUR
            # ------------------------------------------------

            roi = hsv[
                y:y + h,
                x:x + w
            ]


            if roi.size == 0:

                continue


            mean_hue = np.mean(
                roi[:, :, 0]
            )


            if (
                mean_hue < 15
                or mean_hue > 165
            ):

                sign_colour = "Red"


            elif 90 <= mean_hue <= 135:

                sign_colour = "Blue"


            else:

                sign_colour = "Unknown"


            # ------------------------------------------------
            # ESTIMATE SIGN TYPE
            # ------------------------------------------------

            if sign_colour == "Red":

                if shape == "Circle":

                    sign_type = "Possible Speed/Restriction Sign"

                elif shape == "Triangle":

                    sign_type = "Possible Warning Sign"

                else:

                    sign_type = "Possible Regulatory Sign"


            elif sign_colour == "Blue":

                if shape == "Circle":

                    sign_type = "Possible Mandatory Sign"

                else:

                    sign_type = "Possible Information Sign"


            else:

                sign_type = "Traffic Sign"


            # ------------------------------------------------
            # SAVE DETECTION
            # ------------------------------------------------

            detections.append({

                "x": x,
                "y": y,
                "w": w,
                "h": h,

                "area": area,

                "shape": shape,

                "colour": sign_colour,

                "type": sign_type

            })


        return detections


    # ========================================================
    # DRAW DETECTIONS
    # ========================================================

    def draw_detections(
        self,
        frame,
        detections
    ):

        for detection in detections:

            x = detection["x"]
            y = detection["y"]
            w = detection["w"]
            h = detection["h"]

            shape = detection["shape"]
            colour = detection["colour"]
            sign_type = detection["type"]


            # ------------------------------------------------
            # BOUNDING BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 255, 0),
                2
            )


            # ------------------------------------------------
            # LABEL
            # ------------------------------------------------

            label = (
                f"{colour} "
                f"{shape}"
            )


            cv2.putText(
                frame,
                label,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2
            )


            # ------------------------------------------------
            # TYPE
            # ------------------------------------------------

            cv2.putText(
                frame,
                sign_type,
                (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1
            )


        return frame
