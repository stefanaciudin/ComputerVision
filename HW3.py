import os

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt


def rgb_skin_detection(image):
    r, g, b = image[:, :, 2], image[:, :, 1], image[:, :, 0]
    skin_mask = (
        (r > 95) & (g > 40) & (b > 20) &                          # r > 95, g > 40, b > 20
        ((np.max(image, axis=2) - np.min(image, axis=2)) > 15) &  # max{r,g,b} - min{r,g,b} > 15
        (np.abs(r - g) > 15) &                                    # |r - g| > 15
        (r > g) & (r > b)                                         # r > g and r > b
    )
    return skin_mask.astype(np.uint8) * 255

def hsv_skin_detection(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    skin_mask = (
            (hsv_image[:, :, 0] >= 0) & (hsv_image[:, :, 0] <= 25) &
            (hsv_image[:, :, 1] >= 0.23 * 255) & (hsv_image[:, :, 1] <= 0.68 * 255) &
            (hsv_image[:, :, 2] >= 0.35 * 255) & (hsv_image[:, :, 2] <= 255)
    )
    return skin_mask.astype(np.uint8) * 255


def ycbcr_skin_detection(image):
    r, g, b = image[:, :, 2].astype(float), image[:, :, 1].astype(float), image[:, :, 0].astype(float)

    # Apply RGB to YCbCr conversion
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 128

    # Define skin detection criteria in YCbCr space
    skin_mask = (
            (y > 80) &  # Y > 80
            (cb > 85) & (cb < 135) &  # 85 < Cb < 135
            (cr > 135) & (cr < 180)  # 135 < Cr < 180
    )
    return skin_mask.astype(np.uint8) * 255


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



folder_path_a = "lab3/a"
process_images_in_folder(folder_path_a)


def evaluate_skin_detection(prediction, ground_truth):
    # Flatten the arrays to 1D for comparison
    y_true = ground_truth.flatten()
    y_pred = prediction.flatten()
    # Calculate confusion matrix and extract TP, FN, FP, TN
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 255]).ravel()
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn, "accuracy": accuracy}


def display_confusion_matrix(metrics, method_name, filename):
    matrix = np.array([[metrics['TP'], metrics['FN']],
                       [metrics['FP'], metrics['TN']]])
    fig, ax = plt.subplots()
    cax = ax.matshow(matrix, cmap="coolwarm")
    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted Skin', 'Predicted Non-skin'])
    ax.set_yticklabels(['Actual Skin', 'Actual Non-skin'])
    plt.title(f"Confusion Matrix for {method_name.upper()} - {filename}")
    plt.xlabel("Prediction")
    plt.ylabel("Ground Truth")
    for (i, j), val in np.ndenumerate(matrix):
        ax.text(j, i, f"{val}", ha='center', va='center', color='black')
    plt.show()

# Function to process images and evaluate each method
def evaluate_pratheepan_dataset(dataset_path, ground_truth_path):
    methods = {
        "rgb": rgb_skin_detection,
        "hsv": hsv_skin_detection,
        "ycbcr": ycbcr_skin_detection
    }

    results = {method: [] for method in methods}

    for filename in os.listdir(dataset_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(dataset_path, filename)
            gt_path = os.path.join(ground_truth_path, filename)

            image = cv2.imread(image_path)
            ground_truth = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if ground_truth is None:
                print(f"Ground truth for {filename} not found.")
                continue

            for method_name, detection_method in methods.items():
                prediction = detection_method(image)
                metrics = evaluate_skin_detection(prediction, ground_truth)
                results[method_name].append(metrics)

                print(f"Results for {filename} using {method_name.upper()}:")
                print(f"  TP: {metrics['TP']}, FN: {metrics['FN']}, FP: {metrics['FP']}, TN: {metrics['TN']}")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")

                # Display the confusion matrix
                display_confusion_matrix(metrics, method_name, filename)

    for method_name, metrics_list in results.items():
        avg_accuracy = np.mean([m['accuracy'] for m in metrics_list])
        avg_tp = np.mean([m['TP'] for m in metrics_list])
        avg_fn = np.mean([m['FN'] for m in metrics_list])
        avg_fp = np.mean([m['FP'] for m in metrics_list])
        avg_tn = np.mean([m['TN'] for m in metrics_list])

        print(f"\nAverage results for {method_name.upper()}:")
        print(f"  TP: {avg_tp:.1f}, FN: {avg_fn:.1f}, FP: {avg_fp:.1f}, TN: {avg_tn:.1f}")
        print(f"  Average Accuracy: {avg_accuracy:.4f}")


# Paths for the Pratheepan dataset and ground truth
dataset_path = "lab3/b/Pratheepan_Dataset/FacePhoto"
ground_truth_path = "lab3/b/Ground_Truth/GroundT_FacePhoto"

evaluate_pratheepan_dataset(dataset_path, ground_truth_path)


def find_face_bounding_box(image, skin_mask):
    # Find contours in the skin mask
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No skin regions detected.")
        return None

    # Find the largest contour, assumed to be the face region
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate the bounding box around the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    return x, y, w, h


def detect_face_in_image(image_path, method="rgb"):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image at {image_path} could not be loaded.")
        return

    # Apply skin detection method to create skin mask
    if method == "rgb":
        skin_mask = rgb_skin_detection(image)
    else:
        raise ValueError("Currently only RGB method is implemented.")

    # Find the bounding box for the largest skin region
    bounding_box = find_face_bounding_box(image, skin_mask)
    if bounding_box:
        x, y, w, h = bounding_box
        # Draw the bounding box on the original image
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Detected Face with Bounding Box")
        plt.show()
    else:
        print("Face not detected.")

image_path = "lab3/b/Pratheepan_Dataset/FacePhoto/m(01-32)_gr.png"
detect_face_in_image(image_path, method="rgb")
