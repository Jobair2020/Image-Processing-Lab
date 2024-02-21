import cv2
import convolution as co
import kernels as g
import numpy as np
import math


def gaussianFilter():
    print("Select 1 for grayscale image 2 for color 3 for hsv:")
    select = int(input())
    if select == 1:
        img = cv2.imread('Lena.jpg', cv2.IMREAD_GRAYSCALE)
        print("Enter the height and weight for the kernel use odd value")
        height = int(input())
        width = int(input())
        print("Enter the sigma x and sigma y value respectively")
        sigmaX = int(input())
        sigmaY = int(input())
        kernel = g.gaussian(height=height, width=width, sigmaX=sigmaX, sigmaY=sigmaY)
        print('enter center index for the kernel ')
        p = int(input())
        q = int(input())
        co.convolution("gaussian filter", kernel, img, p, q)

    elif select == 2:
        img = cv2.imread('Lena.jpg')
        cv2.imshow("input", img)
        b1, g1, r1 = cv2.split(img)
        print("Enter the height and weight for the kernel use odd value")
        height = int(input())
        width = int(input())
        print("Enter the sigma x and sigma y value respectively")
        sigmaX = int(input())
        sigmaY = int(input())
        kernel = g.gaussian(height=height, width=width, sigmaX=sigmaX, sigmaY=sigmaY)
        print('enter center index of kernel ')
        p = int(input())
        q = int(input())

        b1 = co.convolution("blue", kernel=kernel, img=b1, p=p, q=q)
        g1 = co.convolution("green", kernel=kernel, img=g1, p=p, q=q)
        r1 = co.convolution("red", kernel=kernel, img=r1, p=p, q=q)
        merged = cv2.merge((b1, g1, r1))
        cv2.imshow("gaussian filter", merged)
    else:
        img = cv2.imread('Lena.jpg')
        cv2.imshow("input", img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        b1, g1, r1 = cv2.split(img)
        print("Enter the height and weight for the kernel use odd value")
        height = int(input())
        width = int(input())
        print("Enter the sigma x and sigma y value respectively")
        sigmaX = int(input())
        sigmaY = int(input())
        kernel = g.gaussian(height=height, width=width, sigmaX=sigmaX, sigmaY=sigmaY)
        print('enter center index of kernel ')
        p = int(input())
        q = int(input())

        b1 = co.convolution("blue", kernel=kernel, img=b1, p=p, q=q)
        g1 = co.convolution("green", kernel=kernel, img=g1, p=p, q=q)
        r1 = co.convolution("red", kernel=kernel, img=r1, p=p, q=q)
        merged = cv2.merge((b1, g1, r1))
        cv2.imshow("gaussian hsv", merged)
        rgb = cv2.cvtColor(merged, cv2.COLOR_HSV2RGB)
        cv2.imshow("gaussian rgb", rgb)
        diff = rgb - merged
        cv2.normalize(diff, diff, 0, 255, cv2.NORM_MINMAX)
        diff = np.round(diff).astype(np.uint8)
        cv2.imshow("difference", diff)

    return 0


def meanFilter():
    print("Select 1 for grayscale image 2 for color 3 for hsv:")
    select = int(input())
    if select == 1:
        img = cv2.imread('noisy_image.jpg', cv2.IMREAD_GRAYSCALE)
        print("Enter the height and weight for the kernel use odd value")
        height = int(input())
        width = int(input())
        kernel = g.mean(height=height, width=width)
        print('enter center index for the kernel ')
        p = int(input())
        q = int(input())
        co.convolution("mean filter", kernel, img, p, q)
        cv2.waitKey(0)

    elif select == 2:
        img = cv2.imread('noisy_image.jpg')
        b1, g1, r1 = cv2.split(img)
        print("Enter the height and weight for the kernel use odd value")
        height = int(input())
        width = int(input())
        kernel = g.mean(height=height, width=width)

        print('enter center index of kernel ')
        p = int(input())
        q = int(input())

        b1 = co.convolution("blue", kernel=kernel, img=b1, p=p, q=q)
        g1 = co.convolution("green", kernel=kernel, img=g1, p=p, q=q)
        r1 = co.convolution("red", kernel=kernel, img=r1, p=p, q=q)
        merged = cv2.merge((b1, g1, r1))
        cv2.imshow("mean filter", merged)
        cv2.waitKey(0)

    else:
        img = cv2.imread('noisy_image.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        b1, g1, r1 = cv2.split(img)
        print("Enter the height and weight for the kernel use odd value")
        height = int(input())
        width = int(input())
        kernel = g.mean(height=height, width=width)

        print('enter center index of kernel ')
        p = int(input())
        q = int(input())

        b1 = co.convolution("blue", kernel=kernel, img=b1, p=p, q=q)
        g1 = co.convolution("green", kernel=kernel, img=g1, p=p, q=q)
        r1 = co.convolution("red", kernel=kernel, img=r1, p=p, q=q)
        merged = cv2.merge((b1, g1, r1))
        cv2.imshow("mean filter", merged)
        rgb = cv2.cvtColor(merged, cv2.COLOR_HSV2RGB)
        cv2.imshow("mean rgb", rgb)
        diff = rgb - merged
        cv2.normalize(diff, diff, 0, 255, cv2.NORM_MINMAX)
        diff = np.round(diff).astype(np.uint8)
        cv2.imshow("difference", diff)
        cv2.waitKey(0)

    return 0


def laplacianFilter():
    print("Select 1 for grayscale image 2 for color:")
    select = int(input())
    if select == 1:
        img = cv2.imread('Lena.jpg', cv2.IMREAD_GRAYSCALE)
        cv2.imshow("input", img)
        print("enter the size of kernel")
        size = int(input())
        # size = 5
        kernel = g.laplacian(size, True)
        print('enter center index for kernel ')
        p = int(input())
        q = int(input())

        co.convolution("laplacian of gaussian", kernel, img, p, q)

    else:
        img = cv2.imread('Lena.jpg')
        cv2.imshow("input", img)
        b1, g1, r1 = cv2.split(img)
        print("enter the size of kernel")
        size = int(input())

        kernel = g.laplacian(size, True)

        print('enter center index of kernel ')
        p = int(input())
        q = int(input())

        b1 = co.convolution("blue", kernel=kernel, img=b1, p=p, q=q)
        g1 = co.convolution("green", kernel=kernel, img=g1, p=p, q=q)
        r1 = co.convolution("red", kernel=kernel, img=r1, p=p, q=q)
        merged = cv2.merge((b1, g1, r1))
        cv2.imshow("laplacian filter", merged)
        cv2.waitKey(0)

    return 0


def laplacianOfGaussianFilter():
    print("Select 1 for grayscale image 2 for color:")
    select = int(input())
    if select == 1:
        img = cv2.imread('Lena.jpg', cv2.IMREAD_GRAYSCALE)
        cv2.imshow("input", img)
        print("enter the size of kernel")
        size = int(input())
        # size = 5
        sigma = 1
        kernel = g.laplacian_of_gaussian_kernel(size, sigma)
        print('enter center index for kernel ')
        p = int(input())
        q = int(input())
        co.convolution("laplacian of gaussian", kernel, img, p, q)

    else:
        img = cv2.imread('Lena.jpg')
        cv2.imshow("input", img)
        b1, g1, r1 = cv2.split(img)
        print("enter the size of kernel")
        size = int(input())
        sigma = 1
        kernel = g.laplacian_of_gaussian_kernel(size, sigma)

        print('enter center index of kernel ')
        p = int(input())
        q = int(input())

        b1 = co.convolution("blue", kernel=kernel, img=b1, p=p, q=q)
        g1 = co.convolution("green", kernel=kernel, img=g1, p=p, q=q)
        r1 = co.convolution("red", kernel=kernel, img=r1, p=p, q=q)
        merged = cv2.merge((b1, g1, r1))
        cv2.imshow("laplacian of Gaussian", merged)

    cv2.waitKey(0)
    return 0


def sobelFilter():
    print("Select 1 for grayscale image 2 for color:")
    select = int(input())
    if select == 1:
        img = cv2.imread('Lena.jpg', cv2.IMREAD_GRAYSCALE)
        print('enter center for 3x3 kernel ')
        p = int(input())
        q = int(input())
        kernel_horizontal = g.sobel(True)  # horizontal
        horizontal = co.convolution("sobel_horizontal", kernel_horizontal, img, p, q)
        cv2.waitKey(0)
        kernel_vertical = g.sobel(False)  # vertical
        vertical = co.convolution("sobel_vertical", kernel_vertical, img, p, q)

    else:
        img = cv2.imread('Lena.jpg')
        b1, g1, r1 = cv2.split(img)
        print('enter center for 3x3 kernel ')
        p = int(input())
        q = int(input())
        kernel_horizontal = g.sobel(True)  # horizontal
        b1 = co.convolution("blue", kernel=kernel_horizontal, img=b1, p=p, q=q)
        g1 = co.convolution("green", kernel=kernel_horizontal, img=g1, p=p, q=q)
        r1 = co.convolution("red", kernel=kernel_horizontal, img=r1, p=p, q=q)
        merged = cv2.merge((b1, g1, r1))
        cv2.imshow("horizontal merged", merged)
        cv2.waitKey(0)
        kernel_vertical = g.sobel(False)  # vertical
        b1 = co.convolution("blue", kernel=kernel_vertical, img=b1, p=p, q=q)
        g1 = co.convolution("green", kernel=kernel_vertical, img=g1, p=p, q=q)
        r1 = co.convolution("red", kernel=kernel_vertical, img=r1, p=p, q=q)
        merged = cv2.merge((b1, g1, r1))
        cv2.imshow("vertical merged", merged)

    # gradient_magnitude = np.sqrt(horizontal ** 2 + vertical ** 2)
    # img = np.clip(gradient_magnitude, 0, 255).astype(np.uint8)
    # cv2.imshow('combined image', img)

    return 0


# main
while (True):
    print("Select the type of filter: ")
    print("1  Gaussian filter")
    print("2  Mean Filter")
    print("3  Laplacian Filter")
    print("4  LoG Filter")
    print("5  Sobel Filter")

    choise = int(input())
    if choise == 1:
        gaussianFilter()
    elif choise == 2:
        meanFilter()
    elif choise == 3:
        laplacianFilter()
    elif choise == 4:
        laplacianOfGaussianFilter()
    else:
        sobelFilter()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
