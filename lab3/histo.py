import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("histogram.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("input",img)

#cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
histr = cv2.calcHist([img],[0],None,[256],[0,256])

histogram = np.zeros(256, dtype=np.uint32)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        pixel = image[i, j]
        histogram[pixel] += 1
plt.figure(1)
plt.plot(his)
plt.show()

#plt.figure(figsize=(10, 4))

#plt.subplot(1, 2, 1)
plt.figure(2)
plt.title("Input Image Histogram")
plt.hist(img.ravel(),256,[0,255])
plt.show()

img2 = cv2.equalizeHist(img)

#plt.subplot(1, 2, 2)
plt.figure(3)
plt.title("output Image Histogram")
plt.hist(img2.ravel(),256,[0,255])
plt.show()

cv2.imshow("output",img2)

cv2.waitKey(0)
cv2.destroyAllWindows()