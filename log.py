import numpy as np


def gaussian_kernel(size, sigma):
    """
    Generates a 2D Gaussian kernel.

    Args:
    - size: Size of the kernel (odd integer).
    - sigma: Standard deviation of the Gaussian kernel.

    Returns:
    - 2D Gaussian kernel.
    """
    kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
        -((x - size // 2) ** 2 + (y - size // 2) ** 2) / (2 * sigma ** 2)), (size, size))
    return kernel / np.sum(kernel)


def laplacian_of_gaussian_kernel(size, sigma):
    """
    Generates a Laplacian of Gaussian (LoG) kernel.

    Args:
    - size: Size of the kernel (odd integer).
    - sigma: Standard deviation of the Gaussian kernel.

    Returns:
    - LoG kernel.
    """
    # Generate 2D Gaussian kernel
    gaussian = gaussian_kernel(size, sigma)

    # Compute Laplacian of Gaussian (LoG) kernel
    kernel = np.zeros((size, size))
    for x in range(size):
        for y in range(size):
            kernel[x, y] = -1 * ((x - size // 2) ** 2 + (y - size // 2) ** 2 - 2 * sigma ** 2) * gaussian[x, y]
    return kernel


# Define kernel size and standard deviation
kernel_size = 5
sigma = 1.0

# Generate LoG kernel
LoG_kernel = laplacian_of_gaussian_kernel(kernel_size, sigma)

# Print the LoG kernel
print("Laplacian of Gaussian (LoG) Kernel:")
print(LoG_kernel)


# [[-0.0178141  -0.03991863 -0.04387646 -0.03991863 -0.0178141 ]
#  [-0.03991863 -0.          0.09832033 -0.         -0.03991863]
#  [-0.04387646  0.09832033  0.32420564  0.09832033 -0.04387646]
#  [-0.03991863 -0.          0.09832033 -0.         -0.03991863]
#  [-0.0178141  -0.03991863 -0.04387646 -0.03991863 -0.0178141 ]]