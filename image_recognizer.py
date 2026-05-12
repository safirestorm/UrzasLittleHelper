import cv2
from ultralytics import YOLO
import torchvision.transforms as T
import matplotlib.pyplot as plt

# 1. Load your model
model = YOLO('runs/pose/MTG_Scanner/card_corner_pose-4/weights/best.pt')


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

# 2. Run inference
source_img = "Mtg2-1/test/images/IMG_1321_JPG.rf.d67e35eb58507f3ee5e1bdabfa029583.jpg"
results = model(source_img, conf=0.80)

for r in results:
    if r.keypoints is not None and len(r.keypoints.xy) > 0:
        # Get the first detected card's keypoints as a numpy array
        points = r.keypoints.xy[0].cpu().numpy()

        if len(points) == 4:
            # Warp the image
            flat_card = warp_card(r.orig_img, points)

            print("Warped Result:")
            cv2.imshow("Warped Result: ", pad_to_square(flat_card))
            cv2.waitKey(0)
        else:
            print(f"Detected {len(points)} points. Need exactly 4 to warp.")
    else:
        print("No card keypoints detected.")

cv2.destroyAllWindows()

# Removed cv2.waitKey(0) and cv2.destroyAllWindows() as they crash Colab