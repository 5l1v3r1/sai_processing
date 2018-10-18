import numpy as np
import cv2
from shared import imgs

img = cv2.imread(imgs[1], cv2.IMREAD_COLOR)

'''
It is done with the function, cv2.GaussianBlur(). 
We should specify the width and height of kernel which should be positive and odd.
We also should specify the standard deviation in X and Y direction, sigmaX and sigmaY respectively.
If only sigmaX is specified, sigmaY is taken as same as sigmaX. If both are given as zeros, they are calculated from kernel size.
Gaussian blurring is highly effective in removing gaussian noise from the image.
'''
sigmaX = 5
sigmaY = 5
guasKernel = 0
gausImg = cv2.GaussianBlur(img,(sigmaX,sigmaY),guasKernel)

def gauss_ksize(x):
    guasKernel = x
    gausImg = cv2.GaussianBlur(img, (sigmaX, sigmaY), guasKernel)
    cv2.imshow("GAUSSIAN", gausImg)

def gauss_sigmaX(x):
    sigmaX = x
    gausImg = cv2.GaussianBlur(img, (sigmaX, sigmaY), guasKernel)
    cv2.imshow("GAUSSIAN", gausImg)

def gauss_sigmaY(y):
      sigmaY = y
      gausImg = cv2.GaussianBlur(img, (sigmaX, sigmaY), guasKernel)
      cv2.imshow("GAUSSIAN", gausImg)


'''
Here, the function cv2.medianBlur() takes median of all the pixels under kernel area and 
central element is replaced with this median value. This is highly effective against salt-and-pepper noise in the images.
It reduces the noise effectively. Its kernel size should be a positive odd integer.
'''

median = cv2.medianBlur(img,5)

def median_ksize(x):
    if x % 2 == 0:
        x += 1
    median = cv2.medianBlur(img, x)
    cv2.imshow("MEDIAN", median)


'''
Оператор Собеля — это дискретный дифференциальный оператор, вычисляющий приближение градиента яркости изображения.
Оператор вычисляет градиент яркости изображения в каждой точке. Так находится направление наибольшего увеличения яркости 
и величина её изменения в этом направлении. Результат показывает, насколько «резко» или «плавно» меняется яркость изображения в каждой точке, 
а значит, вероятность нахождения точки на грани, а также ориентацию границы.
Т.о. результатом работы оператора Собеля в точке области постоянной яркости будет нулевой вектор, 
а в точке, лежащей на границе областей различной яркости — вектор, пересекающий границу в направлении увеличения яркости.
Наиболее часто оператор Собеля применяется в алгоритмах выделения границ. 
'''

gray = cv2.imread(imgs[2], cv2.IMREAD_GRAYSCALE)
cv2.imshow("Original", gray)
sobel = cv2.Sobel(gray, ddepth = cv2.CV_64F, dx = 1, dy= 0, ksize= 3)

buf = sobel.copy()
phase = cv2.phase(sobel, buf)
cv2.imshow("Gradient", phase)
sobelSize = 3

def sobel_ksize(x):
    if x % 2 == 0:
        x += 1
    sobelSize = x
    sobel = cv2.Sobel(gray, ddepth= cv2.CV_64F, dx=1, dy=1, ksize=sobelSize)
    buf = sobel.copy()
    phase = cv2.phase(sobel, buf)
    cv2.imshow("Gradient", phase)
    cv2.imshow("SOBEL", sobel)

'''
лапласиан изображения — суммирование производных второго порядка.
'''
laplacian = cv2.Laplacian(img, 	ddepth = 1, ksize = 7)

def laplacian_ksize(x):
    if x % 2 == 0:
        x += 1
    laplacian = cv2.Laplacian(img, ddepth=1, ksize=x)
    cv2.imshow("LAPLACIAN", laplacian)

'''
filter2D
'''

cv2.namedWindow('TrackBars')

#cv2.imshow("ORIGINAL", img)
#cv2.imshow("LAPLACIAN", laplacian)
cv2.imshow("SOBEL", sobel)
#cv2.imshow("MEDIAN", median)
#cv2.imshow("GAUSSIAN", gausImg)

#cv2.createTrackbar("Gauss_KSize", "TrackBars",0,51,gauss_ksize)
#cv2.createTrackbar("Gauss_SX", "TrackBars",0,51,gauss_sigmaX)
#cv2.createTrackbar("Gauss_SY", "TrackBars",0,51,gauss_sigmaY)
#cv2.createTrackbar("Median_KSize", "TrackBars",1,51,median_ksize)
cv2.createTrackbar("Sobel_KSize", "TrackBars",1,25,sobel_ksize)
#cv2.createTrackbar("Laplacian_KSize", "TrackBars",1,25,laplacian_ksize)

cv2.waitKey(0)