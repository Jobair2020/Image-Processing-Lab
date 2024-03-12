import numpy as np
import cv2
import math


def gaussian(sigma=0.7):
    n = int(sigma * 7) | 1
    c = 1 / (2 * 3.1416 * sigma ** 2)
    kernel = np.zeros((n, n))
    n = n // 2
    total = 0

    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            p = -(i * i + j * j) / (2.0 * sigma ** 2)
            g = c * math.exp(p)
            kernel[i + n][j + n] = g
            total += g
    return kernel / total


def derivative_x(kernel, sigma):
    n = kernel.shape[0] // 2
    new = np.zeros((kernel.shape[0], kernel.shape[1]))
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x = (kernel[i + n, j + n] * i) / (sigma ** 2)
            new[i + n, j + n] = x

    return new


def derivative_y(kernel, sigma):
    n = kernel.shape[0] // 2
    new = np.zeros((kernel.shape[0], kernel.shape[1]))
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x = (kernel[i + n, j + n] * j) / (sigma ** 2)
            new[i + n, j + n] = x
    print(new)
    return new


def sobel(img):
    Gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return (Gx, Gy)


# def get_kernel():
#     sigma = 0.7
#     kernel = gaussian(sigma)
#     h = len(kernel)
#     kernel_x = np.zeros((h, h))
#     kernel_y = np.zeros((h, h))
#     mn1 = 100
#     mn2 = 100
#     cx = h // 2
#     for x in range(h):
#         for y in range(h):
#
#             act_x = (x - cx)
#             act_y = (y - cx)
#
#             c1 = -act_x / (sigma ** 2)
#             c2 = -act_y / (sigma ** 2)
#
#             kernel_x[x, y] = c1 * kernel[x, y]
#             kernel_y[x, y] = c2 * kernel[x, y]
#
#             if kernel_x[x, y] != 0:
#                 mn1 = min(abs(kernel_x[x, y]), mn1)
#
#             if kernel_y[x, y] != 0:
#                 mn2 = min(abs(kernel_y[x, y]), mn2)
#
#     dr1 = (kernel_x / mn1).astype(int)
#     dr2 = (kernel_y / mn2).astype(int)
#
#     return (kernel_y, kernel_x)


def globalThresholding(img_gray):
    t = cv2.mean(img_gray)[0]  # Initial threshold estimate
    t_new = 0  # Updated threshold value
    epsilon = 0.001  # Threshold convergence criterion

    while abs(t_new - t) > epsilon:

        mu1 = 0
        mu2 = 0
        count1 = 0
        count2 = 0

        # Update the threshold value
        for i in range(img_gray.shape[0]):
            for j in range(img_gray.shape[1]):
                if img_gray[i, j] > t:
                    mu1 += img_gray[i, j]
                    count1 += 1
                else:
                    mu2 += img_gray[i, j]
                    count2 += 1
        t = t_new  # Update the threshold value for the next iteration
        mu1 /= count1
        mu2 /= count2
        t_new = (mu1 + mu2) / 2

        if abs(t_new - t) <= epsilon:
            break

    return t_new


def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    '''
    Double threshold
    '''
    # t = globalThresholding(img)
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res, weak, strong)


def hysteresis(image, weak, strong=255):
    M, N = image.shape
    out = image.copy()

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if (image[i, j] == weak):
                if np.any(image[i - 1:i + 2, j - 1:j + 2] == strong):
                    out[i, j] = strong
                else:
                    out[i, j] = 0
    return out
