from difflib import SequenceMatcher

import cv2
import numpy as np
from pytesseract import image_to_string
from skimage.util import random_noise

#ground_truth = "This is a lot of 12 point text to test the ocr code and see if it works on all types of file format."
ground_truth = "STARBUCKS Store #10208 11302 Euclid Avenue Cleveland, OH (216) 229-0749 CHK 664290 12/07/2014 06:43 PM 1912003 Drawer: 2 Reg: 2 Vt Pep Mocha 4.95 Sbux Card 4.95 XXXXXXXXXXXX3228 Subtotal $4.95 Total $4.95 Change Due $0.00 Check Closed 12/07/2014 06:43 PM SBUX Card x3228 New Balance: 37.45 Card is registered."


#image_path = 'lab4/testocr-gt.jpg'
image_path = 'lab4/sample21.jpg'

original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def evaluate_ocr(image):
    ocr_text = image_to_string(image, config='--psm 6')
    matcher = SequenceMatcher(None, ocr_text, ground_truth)
    accuracy = matcher.ratio() * 100  # Accuracy as a percentage
    correct_characters = int(matcher.ratio() * len(ground_truth))
    incorrect_characters = len(ground_truth) - correct_characters
    return correct_characters, incorrect_characters, accuracy, ocr_text

# Apply transformations and evaluate each one
results = {}

# 0. Original results

results['Original Image'] = evaluate_ocr(original_image)

# 1. Add Gaussian Noise
noisy_image_gaussian = (255 * random_noise(original_image, mode='gaussian')).astype(np.uint8)
results['Gaussian Noise'] = evaluate_ocr(noisy_image_gaussian)

# 2. Add Salt and Pepper Noise
noisy_image_sp = (255 * random_noise(original_image, mode='s&p', amount=0.1)).astype(np.uint8)
results['Salt and Pepper Noise'] = evaluate_ocr(noisy_image_sp)

# 3. Rotation (5 and 15 degrees)
def rotate_image(image, angle):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))

rotated_image = rotate_image(original_image, 5)
results['Rotation (5 degrees)'] = evaluate_ocr(rotated_image)

another_rotated_image = rotate_image(original_image, 15)
results['Rotation (15 degrees)'] = evaluate_ocr(another_rotated_image)

# 4. Horizontal Shear
def shear_image(image, shear_factor):
    M = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=float)
    return cv2.warpAffine(image, M, (image.shape[1] + int(image.shape[0] * abs(shear_factor)), image.shape[0]))

sheared_image = shear_image(original_image, 0.2)
results['Horizontal Shear'] = evaluate_ocr(sheared_image)

# 5. Resize (Shrink by 50%)
resized_image_shrink = cv2.resize(original_image, (0, 0), fx=0.5, fy=0.5)
results['Resize (50% Shrink)'] = evaluate_ocr(resized_image_shrink)

# 6. Resize (Stretch Vertically)
resized_aspect_image = cv2.resize(original_image, (original_image.shape[1], int(original_image.shape[0] * 1.5)))
results['Resize (Stretch Vertically)'] = evaluate_ocr(resized_aspect_image)

# 7. Average Blur
blurred_avg = cv2.blur(original_image, (5, 5))
results['Average Blur'] = evaluate_ocr(blurred_avg)

# 8. Gaussian Blur
blurred_gaussian = cv2.GaussianBlur(original_image, (5, 5), sigmaX=1)
results['Gaussian Blur'] = evaluate_ocr(blurred_gaussian)

# 9. Image Sharpening
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened_image = cv2.filter2D(original_image, -1, kernel)
results['Sharpening'] = evaluate_ocr(sharpened_image)

# 10. Morphological Dilation
dilated_image = cv2.dilate(original_image, np.ones((3, 3), np.uint8), iterations=1)
results['Dilation'] = evaluate_ocr(dilated_image)

# 11. Morphological Erosion
eroded_image = cv2.erode(original_image, np.ones((3, 3), np.uint8), iterations=1)
results['Erosion'] = evaluate_ocr(eroded_image)

# 12. Thresholding
_, thresholded_image = cv2.threshold(original_image, 128, 255, cv2.THRESH_BINARY)
results['Thresholding'] = evaluate_ocr(thresholded_image)

# Sort results by accuracy in descending order
sorted_results = sorted(results.items(), key=lambda x: x[1][2], reverse=True)

# Print the results for each transformation
print("OCR Results for Each Transformation:\n")
for transformation, (correct, incorrect, accuracy, ocr_text) in sorted_results:
    print(f"Transformation: {transformation}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Correct characters: {correct}")
    print(f"Incorrect characters: {incorrect}")
    print(f"OCR Result:\n{ocr_text}")
    print("-" * 40)

# Print the top 5 transformations with the highest accuracy
print("\nTop 5 Transformations with Highest OCR Accuracy:\n")
for transformation, (correct, incorrect, accuracy, _) in sorted_results[:5]:
    print(f"{transformation}: {accuracy:.2f}% accuracy")

# Print the last 5 transformations with the lowest accuracy
print("\nBottom 5 Transformations with Lowest OCR Accuracy:\n")
for transformation, (correct, incorrect, accuracy, _) in sorted_results[-5:]:
    print(f"{transformation}: {accuracy:.2f}% accuracy")
