import numpy as np
import cv2
import kernels as k


# out = np.zeros((512,512)) #, dtype=np.uint8)
# print(img.max())
# print(img.min())


def convolution(img, kernel):
    n = (kernel.shape[0] // 2)
    row = img.shape[0]
    col = img.shape[1]
    out = np.zeros((row, col))

    img = cv2.copyMakeBorder(src=img, top=n, bottom=n, left=n, right=n, borderType=cv2.BORDER_CONSTANT)
    for i in range(n, row - n):
        for j in range(n, col - n):
            res = 0
            for x in range(-n, n):
                for y in range(-n, n):
                    res += kernel[x + n, y + n] * img.item(i - x, j - y)
            out[i, j] = res

    # print(out)

    out = out[n: -n, n: -n]
    return out


def normalize(out):
    cv2.normalize(out, out, 0, 255, cv2.NORM_MINMAX)
    out = np.round(out).astype(np.uint8)
    return out


def non_maximum_suppression(image, angle):
    image = image.copy()
    image = image / image.max() * 255
    out = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            q = 0
            r = 0
            ang = angle[i, j]
            if (-22.5 <= ang < 22.5) or (157.5 <= ang <= 180) or (-180 <= ang <= -157.5):
                r = image[i, j - 1]
                q = image[i, j + 1]
            elif (-67.5 <= ang <= -22.5) or (112.5 <= ang <= 157.5):
                r = image[i - 1, j + 1]
                q = image[i + 1, j - 1]
            elif (67.5 <= ang <= 112.5) or (-112.5 <= ang <= -67.5):
                r = image[i - 1, j]
                q = image[i + 1, j]
            elif (22.5 <= ang < 67.5) or (-167.5 <= ang <= -112.5):
                r = image[i + 1, j + 1]
                q = image[i - 1, j - 1]

            if (image[i, j] >= q) and (image[i, j] >= r):
                out[i, j] = image[i, j]
            else:
                out[i, j] = 0

    return out


def non_maximum_suppression2(image, angle):
    M, N = image.shape
    Z = np.zeros((M, N), dtype=np.int32)  # resultant image
    # angle = theta * 180. / np.pi  # max -> 180, min -> -180
    angle[angle < 0] += 180  # max -> 180, min -> 0

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q = 255
            r = 255

            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                r = image[i, j - 1]
                q = image[i, j + 1]

            elif (22.5 <= angle[i, j] < 67.5):
                r = image[i - 1, j + 1]
                q = image[i + 1, j - 1]

            elif (67.5 <= angle[i, j] < 112.5):
                r = image[i - 1, j]
                q = image[i + 1, j]

            elif (112.5 <= angle[i, j] < 157.5):
                r = image[i + 1, j + 1]
                q = image[i - 1, j - 1]

            if (image[i, j] >= q) and (image[i, j] >= r):
                Z[i, j] = image[i, j]
            else:
                Z[i, j] = 0

    return out


# main
img = cv2.imread('Lena.jpg', cv2.IMREAD_GRAYSCALE)
sigma = 0.7

kernel = k.gaussian(sigma)

img_conv = convolution(img, kernel)

kernel_x, kernel_y = k.get_kernel()
# kernel_x = k.derivative_x(kernel,sigma)
# kernel_y = k.derivative_y(kernel,sigma)

Gx = convolution(img_conv, kernel_x)
Gy = convolution(img_conv, kernel_y)

Gx = convolution(Gx, kernel)
Gy = convolution(Gy, kernel)

mag = np.sqrt(Gx ** 2 + Gy ** 2)
angle = np.arctan2(Gy, Gx) * 180 / np.pi

cv2.imshow("magnitude", normalize(mag))
cv2.imshow("angel", normalize(angle))

nomaxsup = non_maximum_suppression(mag, angle)

cv2.imshow("non maximum suppression", normalize(nomaxsup))

# t = k.globalThresholding(nomaxsup)

res, weak, strong = k.threshold(nomaxsup)
cv2.imshow("double thresholding", normalize(res))
out = k.hysteresis(res, weak, strong)
cv2.imshow("hysteresis", normalize(out))

cv2.waitKey(0)
cv2.destroyAllWindows()
