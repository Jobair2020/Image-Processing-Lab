import numpy as np
import cv2

img = cv2.imread('box.jpg', cv2.IMREAD_GRAYSCALE)

# img_bordered = cv2.copyMakeBorder(src=img, top=2, bottom=2, left=2, right=2, borderType=cv2.BORDER_CONSTANT)
cv2.imshow('input image', img)

# out = np.zeros((img.shape[0]-1,img.shape[1]-1)) #, dtype=np.uint8)
# out = np.zeros_like(img)
print(img)
# print(img.max())
# print(img.min())

kernel = (1 / 273) * np.array([[1, 4, 7, 4, 1],
                               [4, 16, 26, 16, 4],
                               [7, 26, 41, 26, 7],
                               [4, 16, 26, 16, 4],
                               [1, 4, 7, 4, 1]])
n = (kernel.shape[0] // 2)

img_bordered = cv2.copyMakeBorder(src=img, top=n, bottom=n, left=n, right=n, borderType=cv2.BORDER_CONSTANT)
row = img_bordered.shape[0]
col = img_bordered.shape[1]
# out = img_bordered.copy()
out = np.zeros((row, col))
cv2.imshow('bordered image', img_bordered)
# cv2.imshow('output image', out)
for i in range(n, row - n):
    for j in range(n, col - n):
        res = 0
        for x in range(-n, n + 1):
            for y in range(-n, n + 1):
                f = kernel[x + n, y + n]
                ii = img_bordered.item(i - x, j - y)
                res += f * ii
        out[i, j] = res

# print(out)

print(img_bordered.shape)
print(out.shape)

# cv2.waitKey(0)
cv2.imshow('output image', out)
# print(out)
cv2.normalize(out, out, 0, 255, cv2.NORM_MINMAX)
out = np.round(out).astype(np.uint8)
print(out)
cv2.imshow('normalised output image', out)

cv2.waitKey(0)
cv2.destroyAllWindows()
