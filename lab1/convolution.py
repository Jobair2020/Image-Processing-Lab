import numpy as np
import cv2
import kernels as g


def convolution(s, kernel, img, p=2, q=2):
    k = kernel.shape[0] // 2
    l = kernel.shape[1] // 2

    padding_bottom = kernel.shape[0] - 1 - p
    padding_right = kernel.shape[1] - 1 - q

    img_bordered = cv2.copyMakeBorder(src=img, top=p, bottom=padding_bottom, left=q, right=padding_right,
                                      borderType=cv2.BORDER_CONSTANT)
    out = img_bordered.copy()
    cv2.imshow('bordered image', img_bordered)

    for i in range(p, img_bordered.shape[0] - padding_bottom - k):
        for j in range(q, img_bordered.shape[1] - padding_right - l):
            res = 0
            for x in range(-k, k + 1):
                for y in range(-l, l + 1):
                    res += kernel[x + k, y + l] * img_bordered[i - x, j - y]
            out[i, j] = res

    print(img_bordered.shape)
    print(out.shape)

    # cv2.imshow('output image', out)
    # print(out)
    cv2.normalize(out, out, 0, 255, cv2.NORM_MINMAX)
    out = np.round(out).astype(np.uint8)
    print(f"normalized {out}")
    # crop image to original image
    # out = out[p: -padding_bottom, q:-padding_right]
    cv2.imshow(s, out)
    return out


# main
# img = cv2.imread('Lena.jpg')
# # img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
# cv2.imshow('input image', img)
# b1, g1, r1 = cv2.split(img)
#
# print(img)
#
# kernel_size = 7
# sigma = 1.0
#
# # kernel = g.laplacian_of_gaussian_kernel(kernel_size, sigma)
# # kernel = g.laplacian(kernel_size, True)
# # kernel = g.mean(5,5)
# kernel = g.sobel(False)
#
# # kernel = g.mean(7, 7)
# print(kernel)
# # center
# print('enter center for 5x5 kernel ')
# p = int(input())
# q = int(input())
# img = convolution("sobel", kernel, img, p=2, q=2)
#
# # for color image
# b1 = convolution("blue", kernel=kernel, img=b1, p=p, q=q)
# g1 = convolution("green", kernel=kernel, img=g1, p=p, q=q)
# r1 = convolution("red", kernel=kernel, img=r1, p=p, q=q)
# # merged = cv2.merge((b1, g1, r1))
#
# # cv2.imshow("in hsv", merged)
# # merged = cv2.cvtColor(merged, cv2.COLOR_HSV2RGB)
# # cv2.imshow("merged", merged)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
