import cv2
import numpy as np
from extras import plot_images

"""
ex 1
"""

colored_image = cv2.imread('lab2/sample.jpg')

B, G, R = cv2.split(colored_image)

grayscale_image = (R / 3 + G / 3 + B / 3).astype(np.uint8)
grayscale_image = cv2.merge([grayscale_image, grayscale_image, grayscale_image])
plot_images(["Simple averaging"], [grayscale_image])

"""
ex 2
"""
grayscale_1 = (0.3 * R + 0.59 * G + 0.11 * B).astype(np.uint8)

grayscale_2 = (0.2126 * R + 0.7152 * G + 0.0722 * B).astype(np.uint8)

grayscale_3 = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)

grayscale_1 = cv2.merge([grayscale_1, grayscale_1, grayscale_1])
grayscale_2 = cv2.merge([grayscale_2, grayscale_2, grayscale_2])
grayscale_3 = cv2.merge([grayscale_3, grayscale_3, grayscale_3])

plot_images(["First average", "Second average", "Third average"], [grayscale_1, grayscale_2, grayscale_3])

"""
ex 3
"""

minimum = np.minimum(np.minimum(R, G), B) // 2
maximum = np.maximum(np.maximum(R, G), B) // 2

desaturated_image = (minimum + maximum)

plot_images(["Desaturated"], [cv2.merge([desaturated_image, desaturated_image, desaturated_image])])
"""
ex 4
"""
# Method 1: Maximum decomposition (Gray = max(R, G, B))
grayscale_max = np.maximum(np.maximum(R, G), B).astype(np.uint8)

# Method 2: Minimum decomposition (Gray = min(R, G, B))
grayscale_min = np.minimum(np.minimum(R, G), B).astype(np.uint8)

plot_images(["Maximum decomposition", "Minimum decomposition"],
           [cv2.merge([grayscale_max, grayscale_max, grayscale_max]),
            cv2.merge([grayscale_min, grayscale_min, grayscale_min])])

"""
ex 5
"""
grayscale_red = R.astype(np.uint8)

grayscale_green = G.astype(np.uint8)

grayscale_blue = B.astype(np.uint8)

plot_images(["Red", "Green", "Blue"], [cv2.merge([grayscale_red, grayscale_red, grayscale_red]),
                                       cv2.merge([grayscale_green, grayscale_green, grayscale_green]),
                                       cv2.merge([grayscale_blue, grayscale_blue, grayscale_blue])])

"""
ex 6
"""


def reduce_grayscale_shades(image, num_shades):
    if num_shades > 255 or num_shades < 2:
        raise ValueError("Number of shades must be less than 256 and greater than 2")

    # Convert to grayscale using the weighted average method
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define the size of each interval
    interval_size = 256 // num_shades

    # Create an empty array to store the new grayscale values
    new_gray_values = np.zeros(256, dtype=np.uint8)

    for i in range(num_shades):
        # Define the boundaries for each interval
        lower_bound = i * interval_size
        upper_bound = (i + 1) * interval_size if i < num_shades - 1 else 256

        # Find the average grayscale value of pixels in this interval
        mask = (grayscale >= lower_bound) & (grayscale < upper_bound)
        if np.any(mask):  # Check if there are any pixels in this interval
            avg_value = np.mean(grayscale[mask]).astype(np.uint8)
        else:
            avg_value = (lower_bound + upper_bound - 1) // 2  # Fallback to midpoint if no pixels

        # Assign the average value to all the intensities within the interval
        new_gray_values[lower_bound:upper_bound] = avg_value

    # Apply the mapping to the grayscale image
    reduced_image = new_gray_values[grayscale]

    return reduced_image



num_shades = 4
reduced_gray_image = reduce_grayscale_shades(colored_image, num_shades)

plot_images(["Original", f"Reduced to {num_shades} shades"], [colored_image, cv2.merge([reduced_gray_image,
                                                                                       reduced_gray_image,
                                                                                       reduced_gray_image])])


"""
ex 7
"""


def floyd_steinberg_dithering(image, num_shades):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Rescale grayscale to range 0-255
    height, width = grayscale.shape
    new_image = np.copy(grayscale)

    # Scaling the shades to be between 0 and the number of shades
    scale_factor = 255 // (num_shades - 1)

    for y in range(height):
        for x in range(width):
            old_pixel = new_image[y, x]
            new_pixel = np.round(old_pixel / scale_factor) * scale_factor
            new_image[y, x] = new_pixel
            quant_error = old_pixel - new_pixel

            # Floyd-Steinberg error diffusion
            if x + 1 < width:
                new_image[y, x + 1] += quant_error * 7 / 16
            if y + 1 < height:
                if x - 1 >= 0:
                    new_image[y + 1, x - 1] += quant_error * 3 / 16
                new_image[y + 1, x] += quant_error * 5 / 16
                if x + 1 < width:
                    new_image[y + 1, x + 1] += quant_error * 1 / 16

    return np.clip(new_image, 0, 255).astype(np.uint8)


def stucki_dithering(image, num_shades):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Rescale grayscale to range 0-255
    height, width = grayscale.shape
    new_image = np.copy(grayscale)

    # Scaling the shades to be between 0 and the number of shades
    scale_factor = 255 // (num_shades - 1)

    for y in range(height):
        for x in range(width):
            old_pixel = new_image[y, x]
            new_pixel = np.round(old_pixel / scale_factor) * scale_factor
            new_image[y, x] = new_pixel
            quant_error = old_pixel - new_pixel

            # Stucki error diffusion
            diffusion_matrix = [
                [0, 0, 0, 8 / 42, 4 / 42],
                [2 / 42, 4 / 42, 8 / 42, 4 / 42, 2 / 42],
                [1 / 42, 2 / 42, 4 / 42, 2 / 42, 1 / 42],
            ]

            for i in range(3):
                for j in range(-2, 3):
                    if 0 <= y + i < height and 0 <= x + j < width:
                        new_image[y + i, x + j] += quant_error * diffusion_matrix[i][j + 2]

    return np.clip(new_image, 0, 255).astype(np.uint8)


plot_images(["Original", "Floyd-Steinberg Dithering", "Stucki Dithering"],
            [colored_image, cv2.merge([floyd_steinberg_dithering(colored_image, 4),
                                       floyd_steinberg_dithering(colored_image, 4),
                                       floyd_steinberg_dithering(colored_image, 4)]),
             cv2.merge([stucki_dithering(colored_image, 4),
                        stucki_dithering(colored_image, 4),
                        stucki_dithering(colored_image, 4)])])


def colorize_grayscale_image(image_path, colormap=cv2.COLORMAP_JET):
    # Load the grayscale image
    grayscale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply a colormap to the grayscale image
    colorized_image = cv2.applyColorMap(grayscale_image, colormap)

    # Convert BGR to RGB for plotting with matplotlib
    colorized_image_rgb = cv2.cvtColor(colorized_image, cv2.COLOR_BGR2RGB)

    # Plot the images using plot_images function
    plot_images(["Colorized Image"], [colorized_image_rgb])


colorize_grayscale_image("lab2/rose.jpg", colormap=cv2.COLORMAP_JET)
