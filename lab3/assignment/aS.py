import numpy as np
import cv2
import matplotlib.pyplot as plt


def generate_double_gaussian_hist(size, mean1, std1, mean2, std2):
    x = np.arange(256)
    hist = (
            np.exp(-((x - mean1) ** 2) / (2 * std1 ** 2)) / (std1 * np.sqrt(2 * np.pi)) +
            np.exp(-((x - mean2) ** 2) / (2 * std2 ** 2)) / (std2 * np.sqrt(2 * np.pi))
    )
    hist /= np.sum(hist)
    return hist


def calculate_histogram(image):
    hist = np.zeros(256)
    for pixel in image.flatten():
        hist[pixel] += 1
    return hist / np.sum(hist)


def calculate_cumulative_distribution(hist):
    cumulative_dist = np.zeros_like(hist)
    cumulative_dist[0] = hist[0]
    max_pixel_value = len(hist) - 1  # Maximum pixel value
    for i in range(1, len(hist)):
        cumulative_dist[i] = cumulative_dist[i - 1] + hist[i]
    cumulative_dist *= max_pixel_value  # Multiply by the maximum pixel value
    return cumulative_dist


def histogram_matching(input_image, target_hist):
    input_hist = calculate_histogram(input_image)
    input_cdf = calculate_cumulative_distribution(input_hist)
    target_cdf = calculate_cumulative_distribution(target_hist)

    matched = np.zeros_like(input_image)
    for i in range(256):
        closest_index = find_closest_index(input_cdf[i], target_cdf)
        matched[np.where(input_image == i)] = closest_index

    return matched


def find_closest_index(value, array):
    closest_dist = float('inf')
    closest_index = 0
    for idx, val in enumerate(array):
        dist = abs(value - val)
        if dist < closest_dist:
            closest_dist = dist
            closest_index = idx
    return closest_index


input_image = cv2.imread('histrogram.jpg', cv2.IMREAD_GRAYSCALE)

# User input for mean and standard deviation of the target histogram
mean1 = float(input("Enter mean value for first Gaussian: "))
std1 = float(input("Enter standard deviation for first Gaussian: "))
mean2 = float(input("Enter mean value for second Gaussian: "))
std2 = float(input("Enter standard deviation for second Gaussian: "))

# Generate target histogram (double Gaussian) based on user input
target_hist = generate_double_gaussian_hist(256, mean1, std1, mean2, std2)

# Perform histogram matching
output_image = histogram_matching(input_image, target_hist)

# Display input and output images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Input Image")
plt.imshow(input_image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Output Image")
plt.imshow(output_image, cmap='gray')
plt.show()

# Calculate histograms, PDFs, and CDFs
input_hist = calculate_histogram(input_image)
output_hist = calculate_histogram(output_image)
input_cdf = calculate_cumulative_distribution(input_hist)
output_cdf = calculate_cumulative_distribution(output_hist)

# Plot histograms, PDFs, and CDFs
plt.figure(figsize=(12, 10))

plt.subplot(2, 3, 1)
plt.title("Input Histogram")
plt.bar(range(256), input_hist)

plt.subplot(2, 3, 2)
plt.title("Output Histogram")
plt.bar(range(256), output_hist)

plt.subplot(2, 3, 3)
plt.title("Target Histogram")
plt.plot(target_hist)

plt.subplot(2, 3, 4)
plt.title("Input PDF")
plt.plot(input_hist)

plt.subplot(2, 3, 5)
plt.title("Output PDF")
plt.plot(output_hist)

plt.subplot(2, 3, 6)
plt.title("Target PDF")
plt.plot(target_hist)

plt.tight_layout()
plt.show()