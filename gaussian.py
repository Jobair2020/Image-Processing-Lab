import numpy as np
import cv2
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


cv2.waitKey(0)
cv2.destroyAllWindows()
