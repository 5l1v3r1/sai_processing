import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV

    blur = cv2.GaussianBlur(frame, (15,15), 1)
    median = cv2.medianBlur(frame, 9)




    laplacian = cv2.Laplacian(frame,cv2.CV_64F,ksize=5)
    sobelx = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=5)
    
    cv2.imshow('frame',frame) #params search
    cv2.imshow('gaus',blur) #params search

    cv2.imshow('median',median)



    cv2.imshow('laplacian',laplacian) #params search

    cv2.imshow('sobelx',sobelx) # magnitude
    cv2.imshow('sobely',sobely) # vector


    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
# opencv python example
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_gradients/py_gradients.html
