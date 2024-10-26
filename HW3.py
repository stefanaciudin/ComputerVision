import os

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from skimage import io
import matplotlib.pyplot as plt


def rgb_skin_detection(image):
    # Define skin detection criteria in RGB space
    skin_mask = (
            (image[:, :, 0] > 95) & (image[:, :, 1] > 40) & (image[:, :, 2] > 20) &
            ((np.max(image, axis=2) - np.min(image, axis=2)) > 15) &
            (abs(image[:, :, 0] - image[:, :, 1]) > 15) &
            (image[:, :, 0] > image[:, :, 1]) & (image[:, :, 0] > image[:, :, 2])
    )
    return skin_mask.astype(np.uint8) * 255


def hsv_skin_detection(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    skin_mask = (
            (hsv_image[:, :, 0] >= 0) & (hsv_image[:, :, 0] <= 50) &
            (hsv_image[:, :, 1] >= 0.23 * 255) & (hsv_image[:, :, 1] <= 0.68 * 255) &
            (hsv_image[:, :, 2] >= 0.35 * 255) & (hsv_image[:, :, 2] <= 255)
    )
    return skin_mask.astype(np.uint8) * 255


def ycbcr_skin_detection(image):
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    skin_mask = (
            (ycbcr_image[:, :, 0] > 80) & (ycbcr_image[:, :, 0] <= 255) &  # Y range (80, 255]
            (ycbcr_image[:, :, 1] > 85) & (ycbcr_image[:, :, 1] < 135) &  # Cb range (85, 135)
            (ycbcr_image[:, :, 2] > 135) & (ycbcr_image[:, :, 2] < 180)  # Cr range (135, 180)
    )
    return skin_mask.astype(np.uint8) * 255  # Binary mask with skin pixels as 255 and non-skin as 0


def apply_skin_detection(image, method="rgb"):
    if method == "rgb":
        return rgb_skin_detection(image)
    elif method == "hsv":
        return hsv_skin_detection(image)
    elif method == "ycbcr":
        return ycbcr_skin_detection(image)
    else:
        raise ValueError("Unknown method. Use 'rgb', 'hsv', or 'ycbcr'.")


def display_skin_detection_results(image, skin_masks):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, len(skin_masks) + 1, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for i, (method, mask) in enumerate(skin_masks.items(), start=2):
        plt.subplot(1, len(skin_masks) + 1, i)
        plt.title(f"Skin Detection ({method.upper()})")
        plt.imshow(mask, cmap='gray')
    plt.show()


# Function to process all images in the folder
def process_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing image: {filename}")
            image = cv2.imread(image_path)
            skin_masks = {
                "rgb": apply_skin_detection(image, method="rgb"),
                "hsv": apply_skin_detection(image, method="hsv"),
                "ycbcr": apply_skin_detection(image, method="ycbcr")
            }
            display_skin_detection_results(image, skin_masks)


def load_images(image_path, ground_truth_path):
    image = io.imread(image_path)
    ground_truth = io.imread(ground_truth_path, as_gray=True)  # Ground truth as binary
    return image, (ground_truth > 0).astype(np.uint8) * 255  # Binary mask


def evaluate_skin_detection(prediction, ground_truth):
    # Calculate confusion matrix values
    tn, fp, fn, tp = confusion_matrix(ground_truth.flatten(), prediction.flatten(), labels=[0, 255]).ravel()
    accuracy = accuracy_score(ground_truth.flatten(), prediction.flatten())
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn, "accuracy": accuracy}


folder_path = "lab3/a"
process_images_in_folder(folder_path)
