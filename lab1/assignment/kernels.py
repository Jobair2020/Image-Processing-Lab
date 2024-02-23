import numpy as np
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


def gaussian(sigmaX=1, sigmaY=1):
    # makes it odd
    height = int(sigmaX * 7) | 1
    width = int(sigmaY * 7) | 1
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
            # print(kernel[x + height, y + width], end=" ")
        # print()

    print(kernel / sum)

    return kernel / sum


def mean(height=5, width=6):
    kernel = np.ones((height, width), dtype=np.float32)
    kernel = kernel / (height * width)

    print(kernel)
    return kernel


def laplacian(size=5, centcoff=True):
    coff = 1 if centcoff else -1

    kernel = np.ones((size, size), dtype=int) * coff
    center = size // 2
    kernel[center, center] = ((size * size) - 1) * (- coff)

    print(kernel)
    return kernel


def sobel(horizontal=True):
    kernel_h = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_v = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return kernel_h if horizontal else kernel_v


def laplacian_of_gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size))
    c = (1 / (2 * np.pi * sigma ** 2))
    size = size // 2

    for x in range(-size, size + 1):
        for y in range(-size, size + 1):
            kernel[x + size, y + size] = c * np.exp(-((x * x) + (y * y)) / (2 * sigma ** 2))

    gauss = kernel / np.sum(kernel)

    for x in range(-size, size + 1):
        for y in range(-size, size + 1):
            kernel[x + size, y + size] = -1 * ((x * x) + (y * y) - (2 * sigma ** 2)) * gauss[x + size, y + size]

    print(kernel)

    return kernel

# mean()
# cv2.waitKey(0)
# cv2.destroyAllWindows()
