import cv2
import matplotlib.pyplot as plt
import numpy as np
from extras import plot_images

"""
ex 2
"""

image = cv2.imread('lab1/pictures/lena.tif')

print(f"Image Size (Height, Width, Channels): {image.shape}")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.title('Lena Picture')
plt.axis('off')
plt.show()

cv2.imwrite('lab1/lena_saved.tif', image)

"""
ex 3
"""


# 1. Apply Gaussian Blur with different kernel sizes
blur_5x5 = cv2.GaussianBlur(image_rgb, (5, 5), 0)  # 5x5 kernel
blur_9x9 = cv2.GaussianBlur(image_rgb, (9, 9), 0)  # 9x9 kernel

# 2. Apply sharpening using a custom kernel
# Define a sharpening kernel
sharpen_kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])

# Apply the sharpening filter using cv2.filter2D
sharpened_image = cv2.filter2D(image_rgb, -1, sharpen_kernel)

# Test with a different sharpening kernel for a more subtle effect
sharpen_kernel_subtle = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])

sharpened_image_subtle = cv2.filter2D(image_rgb, -1, sharpen_kernel_subtle)

# Plot the original, blurred, and sharpened images
plot_images(
    titles=['Original', 'Gaussian Blur (5x5)', 'Gaussian Blur (9x9)', 'Sharpened', 'Sharpened (Subtle)'],
    images=[image_rgb, blur_5x5, blur_9x9, sharpened_image, sharpened_image_subtle]
)

cv2.imwrite('lab1/filters/blurred_5x5.tif', cv2.cvtColor(blur_5x5, cv2.COLOR_RGB2BGR))
cv2.imwrite('lab1/filters/blurred_9x9.tif', cv2.cvtColor(blur_9x9, cv2.COLOR_RGB2BGR))
cv2.imwrite('lab1/filters/sharpened_image.tif', cv2.cvtColor(sharpened_image, cv2.COLOR_RGB2BGR))
cv2.imwrite('lab1/filters/sharpened_image_subtle.tif', cv2.cvtColor(sharpened_image_subtle, cv2.COLOR_RGB2BGR))

"""
ex 4
"""

requested_kernel = np.array([[0, -2, 0],
                             [-2, 8, -2],
                             [0, -2, 0]])
image_with_kernel = cv2.filter2D(image_rgb, -1, requested_kernel)
cv2.imwrite('lab1/req_filter/image_with_kernel.tif', cv2.cvtColor(image_with_kernel, cv2.COLOR_RGB2BGR))

"""
ex 5
"""


def rotate_image(image, angle):
    # Get the image dimensions (height and width)
    (h, w) = image.shape[:2]
    # Get the center of the image
    center = (w // 2, h // 2)
    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

    return rotated_image


rotated_clockwise_45 = rotate_image(image_rgb, -45)
rotated_counterclockwise_90 = rotate_image(image_rgb, 90)

plot_images(
    titles=['Original', 'Rotated -45° (Clockwise)', 'Rotated 90° (Counterclockwise)'],
    images=[image_rgb, rotated_clockwise_45, rotated_counterclockwise_90]
)

cv2.imwrite('lab1/rotated/rotated_clockwise_45.tif', cv2.cvtColor(rotated_clockwise_45, cv2.COLOR_RGB2BGR))
cv2.imwrite('lab1/rotated/rotated_counterclockwise_90.tif', cv2.cvtColor(rotated_counterclockwise_90, cv2.COLOR_RGB2BGR))

"""
ex 6
"""


def crop_image(image, x, y, width, height):
    # Ensure that the cropping dimensions are valid
    if (x + width > image.shape[1]) or (y + height > image.shape[0]) or (x < 0) or (y < 0):
        raise ValueError("Crop dimensions are out of image bounds")

    # Crop the image using array slicing
    cropped_image = image[y:y + height, x:x + width]

    return cropped_image


x_start = 50
y_start = 50
crop_width = 200
crop_height = 150

# Perform cropping
cropped_image = crop_image(image_rgb, x_start, y_start, crop_width, crop_height)
plot_images(titles=['Original Image', 'Cropped Image'], images=[image_rgb, cropped_image])

"""
ex 7
"""

height, width = 400, 400
kissing_emoji_image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

center_face = (200, 200)
radius_face = 180
cv2.circle(kissing_emoji_image, center_face, radius_face, (255, 221, 103), -1)

eye_color = (102, 78, 39)
left_eye_center = (140, 170)
right_eye_center = (260, 170)
eye_radius = 25
cv2.circle(kissing_emoji_image, left_eye_center, eye_radius, eye_color, -1)
cv2.circle(kissing_emoji_image, right_eye_center, eye_radius, eye_color, -1)

cv2.ellipse(kissing_emoji_image, (140, 130), (50, 20), 0, 180, 360, eye_color, thickness=10)
cv2.ellipse(kissing_emoji_image, (260, 130), (50, 20), 0, 180, 360, eye_color, thickness=10)

mouth_color = eye_color
pts_mouth = np.array([[190, 260], [210, 250], [200, 270], [220, 280], [210, 290]], np.int32)
cv2.polylines(kissing_emoji_image, [pts_mouth], False, mouth_color, thickness=15)

cv2.imshow('Kissing Face Emoji', kissing_emoji_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('lab1/StefanaCiudin_smiley_face_emoji.png', kissing_emoji_image)
