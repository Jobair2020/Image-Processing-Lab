import numpy as np
import cv2
import math


# img = cv2.imread('lab1/box.jpg', cv2.IMREAD_GRAYSCALE)
# img_bordered = cv2.copyMakeBorder(src=img, top=2, bottom=2, left=2, right=2, borderType=cv2.BORDER_CONSTANT)
# cv2.imshow('grayscaled image', img)
# # cv2.imshow('bordered image', img_bordered)
# out = img.copy()
# out = np.zeros((512,512)) #, dtype=np.uint8)
# print(img.max())
# print(img.min())

# kernel = (1 / 273) * np.array([[1, 4, 7, 4, 1],
#                                [4, 16, 26, 16, 4],
#                                [7, 26, 41, 26, 7],
#                                [4, 16, 26, 16, 4],
#                                [1, 4, 7, 4, 1]])


def gaussian(sigma=0.7, n=7):
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


def convolution(img, kernel):
    n = (kernel.shape[0] // 2)
    row = img.shape[0]
    col = img.shape[1]
    out = np.zeros((row, col))

    for i in range(n, row - n):
        for j in range(n, col - n):
            res = 0
            for x in range(-n, n):
                for y in range(-n, n):
                    res += kernel[x + n, y + n] * img.item(i - x, j - y)
            out[i, j] = res

    print(out)
    # cv2.imshow('normalised output image', out)
    return out


def derivative_x(kernel):
    n = kernel.shape[0] // 2
    new = np.zeros((kernel.shape[0], kernel.shape[1]))
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x = (kernel[i + n, j + n] * i) / (sigma ** 2)
            new[i + n, j + n] = x

    return new


def derivative_y(kernel):
    n = kernel.shape[0] // 2
    new = np.zeros((kernel.shape[0], kernel.shape[1]))
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x = (kernel[i + n, j + n] * j) / (sigma ** 2)
            new[i + n, j + n] = x
    print(new)
    return new


def theholding(img):
    t = cv2.mean(img)
    count1 = 0
    count2 = 0
    mu1 = 0
    mu2 = 0
    t1 = t
    while (t1 != t):
        t1 = t
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i, j] > t:
                    count1 = count1 + 1
                    mu1 += img[i, j]
                else:
                    count2 = count2 + 1
                    mu2 += img[i, j]
        t = (mu1 + mu2) / 2

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] > t:
                img[i, j] = 255
            else:
                img[i, j] = 0

    return img


def normalize(out):
    cv2.normalize(out, out, 0, 255, cv2.NORM_MINMAX)
    out = np.round(out).astype(np.uint8)
    return out


# main
img = cv2.imread('lab1\Lena.jpg', cv2.IMREAD_GRAYSCALE)
size = 7
sigma = 0.7
kernel = gaussian(sigma, size)
border = size // 2
img_bordered = cv2.copyMakeBorder(src=img, top=border, bottom=border, left=border, right=border,
                                  borderType=cv2.BORDER_CONSTANT)

kernel_x = derivative_x(kernel)
out1 = convolution(img_bordered, kernel_x)
out = normalize(out1)
cv2.imshow("x derivative", out)
kernel_y = derivative_y(kernel)
out2 = convolution(img_bordered, kernel_y)
out = normalize(out2)
cv2.imshow("y derivative", out)

combined = img_bordered.copy()
for i in range(border, img_bordered.shape[0] - border):
    for j in range(border, img_bordered.shape[1] - border):
        val = math.sqrt(out1[i, j] ** 2 + out2[i, j] ** 2)
        combined[i, j] = val


out = normalize(combined)
cv2.imshow("magnitude", out)
# out = theholding(combined)

cv2.waitKey(0)
cv2.destroyAllWindows()
