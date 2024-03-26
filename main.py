import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

def find(target, input):
    diff = target - input
    mask = np.ma.less_equal(diff, 0)

    if np.all(mask):
        c = np.abs(diff).argmin()
        return c
    masked_diff = np.ma.masked_array(diff, mask)
    return masked_diff.argmin()

# Step 1: Generate the target histogram using the double Gaussian distribution
x = 256  # Number of bins
mu1, sigma1 = 50, 10  # Parameters for the first Gaussian
mu2, sigma2 = 200, 20  # Parameters for the second Gaussian

# Generate values for x-axis
x_values = np.arange(0, x, 1)

# Calculate the double Gaussian distribution
gaussian1 = (1 / (np.sqrt(2 * np.pi) * sigma1)) * np.exp(-((x_values - mu1) ** 2) / (2 * sigma1 ** 2))
gaussian2 = (1 / (np.sqrt(2 * np.pi) * sigma2)) * np.exp(-((x_values - mu2) ** 2) / (2 * sigma2 ** 2))
target_histogram = gaussian1 + gaussian2

# Normalize the histogram
target_histogram /= np.sum(target_histogram)

# Step 2: Apply histogram matching to update the histogram of the input grayscale image
img = cv2.imread('histogram.jpg', cv2.IMREAD_GRAYSCALE)

# Calculate input histogram
histogram, _ = np.histogram(img.flatten(), 256, [0, 256])
pdf = histogram / float(img.shape[0] * img.shape[1])

# Calculate CDF of input image
cdf = np.cumsum(pdf) * 255

# Calculate CDF of target histogram
target_cdf = np.cumsum(target_histogram) * 255

# Match histograms
output_cdf = np.zeros(256)
for i in range(256):
    output_cdf[i] = find(target_cdf, cdf[i])

# Apply histogram matching
output = output_cdf[img]
output = output.astype(np.uint8)

# Step 3: Show input and output images
plt.figure(figsize=(15, 10))

# Plot input histogram, PDF, and CDF
plt.subplot(3, 3, 1)
plt.plot(histogram, color='blue')
plt.title('Input Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.grid(True)

plt.subplot(3, 3, 2)
plt.plot(pdf, color='green')
plt.title('Input PDF')
plt.xlabel('Pixel Intensity')
plt.ylabel('Probability')
plt.grid(True)

plt.subplot(3, 3, 3)
plt.plot(cdf, color='red')
plt.title('Input CDF')
plt.xlabel('Pixel Intensity')
plt.ylabel('Cumulative Probability')
plt.grid(True)

# Plot target histogram
plt.subplot(3, 3, 4)
plt.plot(target_histogram, color='blue')
plt.title('Target Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Probability')
plt.grid(True)

# Show input image
plt.subplot(3, 3, 5)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Input Image')
plt.axis('off')

# Show output image
plt.subplot(3, 3, 6)
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title('Output Image')
plt.axis('off')

# Plot histogram of output image
plt.subplot(3, 3, 7)
plt.hist(output.flatten(), 256, [0, 256], color='blue')
plt.title('Histogram of Output Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.grid(True)

# Calculate output histogram, PDF, and CDF
output_histogram, _ = np.histogram(output.flatten(), 256, [0, 256])
output_pdf = output_histogram / float(img.shape[0] * img.shape[1])
output_cdf = np.cumsum(output_pdf) * 255

# Plot output PDF and CDF
plt.subplot(3, 3, 8)
plt.plot(output_pdf, color='green')
plt.title('Output PDF')
plt.xlabel('Pixel Intensity')
plt.ylabel('Probability')
plt.grid(True)

plt.subplot(3, 3, 9)
plt.plot(output_cdf, color='red')
plt.title('Output CDF')
plt.xlabel('Pixel Intensity')
plt.ylabel('Cumulative Probability')
plt.grid(True)

# Show all plots
plt.tight_layout()
plt.show()