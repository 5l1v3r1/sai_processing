from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt
import cv2
from shared import imgs

file_name = imgs[0]

img = cv2.imread(file_name, cv2.IMREAD_COLOR)


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype(np.uint8)

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


gammaImg = adjust_gamma(img, 1)


def set_gamma(g):
    if g == 0:
        g = 0.1
    else:
        g /= 25.0
    gamma_img = adjust_gamma(img, g)
    cv2.imshow("GAMMA CORRECTION", gamma_img)


def hand_correction(img, alpha=1.0, beta=0.0):
    buf = img.copy()
    y_size, x_size, c = buf.shape
    for y in range(y_size):
        for x in range(x_size):
            buf[y][x] = buf[y][x] * alpha + beta
    return buf


a = 1
b = 0

handImg = hand_correction(img, a, b)


def set_alpha(alpha):
    alpha /= 25.0
    a = alpha
    hand_img = hand_correction(img, a, b)
    cv2.imshow("HAND_CORRECTION", hand_img)


def set_beta(beta):
    b = beta
    hand_img = hand_correction(img, a, b)
    cv2.imshow("HAND_CORRECTION", hand_img)


# hist = cv2.calcHist([img],[0],None,[256],[0,256])


cv2.namedWindow('TrackBars')

cv2.createTrackbar("G: 0.1 to 2", "TrackBars", 0, 50, set_gamma)
cv2.createTrackbar("HA: 0.1 to 2", "TrackBars", 0, 50, set_alpha)
cv2.createTrackbar("HB: 1 to 50", "TrackBars", 0, 50, set_beta)

imgGray = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
# imgGray = cv2.imread("sobel.jpg", cv2.IMREAD_GRAYSCALE)

cv2.imshow("ORIGINAL", img)
cv2.imshow("ORIGINAL_GREY", imgGray)
cv2.imshow("GAMMA CORRECTION", gammaImg)
cv2.imshow("HAND_CORRECTION", handImg)

##src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
dst = cv2.equalizeHist(imgGray)

cv2.imshow('Equalized Image', dst)

plt.hist(imgGray.ravel(), 256, [0, 256], label='original')

plt.hist(dst.ravel(), 256, [0, 256], label='equalized')
plt.legend(loc="upper left")
plt.show()

cv2.waitKey(0)
