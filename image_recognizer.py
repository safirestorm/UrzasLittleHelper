import cv2
from ultralytics import YOLO
import torchvision.transforms as T
import matplotlib.pyplot as plt

# 1. Load your model
model = YOLO('runs/pose/MTG_Scanner/card_corner_pose-5/weights/best.pt')


import numpy as np


def pad_to_square(image):
    """Adds black padding to make the image square using OpenCV."""
    h, w = image.shape[:2]
    max_side = max(h, w)

    top = (max_side - h) // 2
    bottom = max_side - h - top
    left = (max_side - w) // 2
    right = max_side - w - left

    # Border types: cv2.BORDER_CONSTANT adds a solid color (default black)
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

def warp_card(image, kpts):
    """
    Warps a card image to a top-down view using 4 keypoints.
    kpts: numpy array of shape (4, 2)
    """
    # 1. Sort points to ensure they are in the order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    # We do this by calculating the sum and difference of the coordinates
    s = kpts.sum(axis=1)
    diff = np.diff(kpts, axis=1)

    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = kpts[np.argmin(s)]  # Top-Left (smallest sum)
    rect[2] = kpts[np.argmax(s)]  # Bottom-Right (largest sum)
    rect[1] = kpts[np.argmin(diff)]  # Top-Right (smallest difference)
    rect[3] = kpts[np.argmax(diff)]  # Bottom-Left (largest difference)

    # 2. Define the destination dimensions (Standard MTG ratio)
    width = 630
    height = 880
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype="float32")

    # 3. Calculate the Perspective Transform Matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (width, height))

    return warped


def getWarpedImage(image):
    results = model(image, conf=0.80)
    firstCard = results[0]
    if firstCard.keypoints is not None and len(firstCard.keypoints.xy) > 0:
        points = firstCard.keypoints.xy[0].cpu().numpy()

        if len(points) == 4:
            flat_card = warp_card(firstCard.orig_img, points)
            padded = pad_to_square(flat_card)
            return padded
        else:
            print("Not enough points")
            return None
    else:
        print("No card keypoints detected.")
        return None


def getAnnotatedImage(image):
    results = model(image, conf=0.80)
    result = results[0]

    # Work on a copy of the original image to avoid modifying the source
    annotated_img = result.orig_img.copy()

    if result.boxes is not None and result.keypoints is not None:
        # 1. Draw Bounding Boxes
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green Box

        # 2. Draw Keypoints
        if len(result.keypoints.xy) > 0:
            points = result.keypoints.xy[0].cpu().numpy()
            for (x, y) in points:
                cv2.circle(annotated_img, (int(x), int(y)), 10, (0, 0, 255), -1)  # Red Dots

        return annotated_img

    return None