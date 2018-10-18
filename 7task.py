import cv2 as cv
import numpy as np
from shared import imgs

sLine = 200
pLine = 200
cannyth = 120
fname = "/Users/kolsha/Pictures/2O_aUamZRZU.jpg"


def standard_lines(dst):
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    lines = cv.HoughLines(dst, 1, np.pi / 180, sLine)

    if lines is not None:
        for line in lines:
            rho = line[0][0]
            theta = line[0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
            cv.line(cdst, pt1, pt2, (0, 0, 255), 2)

    cv.imshow("Standard Hough Line Transform", cdst)


def probabal_lines(dst):
    cdstP = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, pLine, None, 50, 10)

    if linesP is not None:
        for line in linesP:
            l = line[0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 2)

    cv.imshow("Probabilistic Line Transform", cdstP)


def canny(val):
    cannyth = val
    filename = fname
    # Loads an image
    src = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    dst = cv.Canny(src, cannyth, 150, apertureSize=3)

    # Copy edges to the images that will display the results in BGR
    standard_lines(dst)
    probabal_lines(dst)
    return src, dst


src, dst = canny(50)


def standard_lines_tb(val):
    global sLine
    sLine = val
    standard_lines(dst)


def probabal_lines_tb(val):
    global pLine
    pLine = val
    probabal_lines(dst)


cv.namedWindow("Thresholds")
cv.createTrackbar("Canny", "Thresholds", 120, 255, canny)
cv.createTrackbar("Standard", "Thresholds", 200, 255, standard_lines_tb)
cv.createTrackbar("Probabalistic", "Thresholds", 200, 255, probabal_lines_tb)

cv.imshow("Source", src)

cv.waitKey()
