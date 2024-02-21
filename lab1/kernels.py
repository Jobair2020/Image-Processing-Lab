import numpy as np
# import cv2
import math

# void cv::GaussianBlur(InputArray src,OutputArray dst,Size ksize,double sigmaX,double sigmaY = 0,int borderType = BORDER_DEFAULT)
# blur = cv2.GaussianBlur(img,(5,5),1)
# kernel = (1 / 273) * np.array([[1, 4, 7, 4, 1],
#                                [4, 16, 26, 16, 4],
#                                [7, 26, 41, 26, 7],
#                                [4, 16, 26, 16, 4],
#                                [1, 4, 7, 4, 1]])
# void cv::filter2D(InputArray src, OutputArray dst,int ddepth,InputArray kernel,Point anchor = Point(-1,-1),double delta = 0,intborderType = BORDER_DEFAULT )
# out = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)


def gaussian(height=5, width=5, sigmaX=1, sigmaY=1):
    # sigmaX = int(input())
    # sigmaY = int(input())
    kernel = np.zeros((height, width))
    height = height // 2
    width = width // 2
    c = (1 / (2 * 3.1416 * sigmaX * sigmaY))
    sigmaX = sigmaX * sigmaX
    sigmaY = sigmaY * sigmaY
    sum = 0
    for x in range(-height, height + 1):
        for y in range(-width, width + 1):
            g = c * math.exp(-0.5 * ((x * x / sigmaX) + (y * y / sigmaY)))

            kernel[x + height, y + width] = "{:.4f}".format(g)
            sum += g
            print(kernel[x + height, y + width], end=" ")
        print()

    return kernel / sum


def mean(height=5, width=6):
    kernel = np.ones((height, width), dtype=np.float32)
    kernel = kernel / (height * width)

    return kernel


def laplacian(size=5, centcoff=True):
    coff = 1 if centcoff else -1

    kernel = np.ones((size, size), dtype=int) * coff
    center = size // 2
    kernel[center, center] = ((size * size) - 1) * (- coff)

    return kernel


def gaussian_kernel(sigma, size):
    kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
        -((x - size // 2) ** 2 + (y - size // 2) ** 2) / (2 * sigma ** 2)), (size, size))
    return kernel / np.sum(kernel)


def laplacian_of_gaussian(sigma, size):
    # a 1D Laplacian kernel
    laplacian_kernel_1d = np.array([1, -2, 1])

    # convolution the Laplacian kernel with the Gaussian kernel
    gaussian_kernel_1d = gaussian_kernel(sigma, size)
    laplacian_of_gaussian_1d = np.convolve(gaussian_kernel_1d, laplacian_kernel_1d, mode='same')

    # the 2D Laplacian of Gaussian kernel
    laplacian_of_gaussian_2d = np.outer(laplacian_of_gaussian_1d, laplacian_of_gaussian_1d)

    return laplacian_of_gaussian_2d


def sobel(h=True):
    kernel_h = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_v = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return kernel_h if h else kernel_v

sigma = 1.4
size = 5

log_kernel = laplacian_of_gaussian(sigma, size)
print("Laplacian of Gaussian Kernel:")
print(log_kernel)

mean()
# cv2.waitKey(0)
# cv2.destroyAllWindows()
