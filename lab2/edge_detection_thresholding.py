import numpy as np
import cv2
import math


# out = np.zeros((512,512)) #, dtype=np.uint8)
# print(img.max())
# print(img.min())

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

def Thresholding(img_gray):
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

    # Apply thresholding to the image
    # ret, img_thresh = cv2.threshold(img_gray, t, 255, cv2.THRESH_BINARY)
    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            if (img_gray[i, j] > t):
                img_gray[i, j] = 255
            else:
                img_gray[i, j] = 0

    return img_gray


def normalize(out):
    cv2.normalize(out, out, 0, 255, cv2.NORM_MINMAX)
    out = np.round(out).astype(np.uint8)
    return out


# main
img = cv2.imread('../lab1/assignment/Lena.jpg', cv2.IMREAD_GRAYSCALE)
sigma = 0.7
kernel = gaussian(sigma)
border = kernel.shape[0] // 2
img_bordered = cv2.copyMakeBorder(src=img, top=border, bottom=border, left=border, right=border,
                                  borderType=cv2.BORDER_CONSTANT)

kernel_x = derivative_x(kernel)
out1 = convolution(img_bordered, kernel_x)
out = out1.copy()
out = normalize(out)
cv2.imshow("x derivative", out)
kernel_y = derivative_y(kernel)
out2 = convolution(img_bordered, kernel_y)
out = out2.copy()
out = normalize(out)
cv2.imshow("y derivative", out)

combined = img_bordered.copy()
for i in range(border, img_bordered.shape[0] - border):
    for j in range(border, img_bordered.shape[1] - border):
        val = math.sqrt(out1[i, j] ** 2 + out2[i, j] ** 2)
        combined[i, j] = val

cv2.imshow("magnitude", combined)

out = Thresholding(combined)
cv2.imshow("thresholding", out)

cv2.waitKey(0)
cv2.destroyAllWindows()
