import cv2
import numpy as np
import matplotlib.pyplot as plt
from shared import imgs

'''
def image_gradients(im, kernel_size):
    grad_x = (cv2.Sobel(im, ddepth=-1, dx=1, dy=0, ksize=kernel_size)).astype(np.float32)
    grad_y = (cv2.Sobel(im, ddepth=-1, dx=0, dy=1, ksize=kernel_size)).astype(np.float32)

    modulus = np.expand_dims(np.sqrt(grad_x * grad_x + grad_y * grad_y) / (127.5 * np.sqrt(2)), axis=3)
    direction = (np.arctan2(grad_x - 127.5, grad_y - 127.5) / np.pi + 1.0) / 2.0
    colormap = plt.get_cmap("hsv")

    direction_image = colormap(direction).astype(np.float32)
    dir_mod_image = modulus * direction_image * 255.0
    result = dir_mod_image[:, :, 0:3]

    return result.astype(np.uint8)

'''
frame = cv2.imread(imgs[10])

# Convert BGR to HSV

blur = cv2.GaussianBlur(frame, (15, 15), 0, 0)
median = cv2.medianBlur(frame, 19)

laplacian = cv2.Laplacian(frame, cv2.CV_64F, ksize=7)

frame = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)

sobelx = cv2.Sobel(frame, ddepth=-1, dx=1, dy=0, ksize=3).astype(np.float32)
sobely = cv2.Sobel(frame, ddepth=-1, dx=0, dy=1, ksize=3).astype(np.float32)

abs_sobel_y = cv2.convertScaleAbs(sobely)
abs_sobel_x = cv2.convertScaleAbs(sobelx)

modulus = np.expand_dims(np.sqrt(sobelx * sobelx + sobely * sobely) / (127.5 * np.sqrt(2)), axis=3)
direction = (np.arctan2(sobelx - 127.5, sobely - 127.5) / np.pi + 1.0) / 2.0
colormap = plt.get_cmap("gist_rainbow")

sobelx = abs_sobel_x
sobely = abs_sobel_y

direction_image = colormap(direction).astype(np.float32)
dir_mod_image = modulus * direction_image * 255.0
result = dir_mod_image[:, :, 0:3]

grd = result.astype(np.uint8)

#cv2.imshow('grd', grd)

treshold = 35.0

for x in range(sobelx.shape[0]):
    for y in range(sobelx.shape[1]):
        bright = sobelx[x, y].sum() / 3.0
        if bright > treshold:
            sobelx[x, y] = 255

# abs_sobel_x = abs_sobel_x * 100
# abs_sobel_y = abs_sobel_y * 100

# sobelx = cv2.addWeighted(sobelx , 0.1, abs_sobel_x, 0.9, 0.0)

# sobely = cv2.addWeighted( sobely, 0.1, abs_sobel_x, 0.9, 0.0)

imgSobelX = cv2.Sobel(frame, -1, 1, 0, ksize=3)
imgSobelY = cv2.Sobel(frame, -1, 0, 1, ksize=3)
module = np.sqrt(np.power(imgSobelX.astype(np.float32), 2) + np.power(imgSobelY.astype(np.float32), 2))
arctan = np.arctan2(imgSobelX, imgSobelY)
direction = (arctan / np.pi + 1.0) / 2.0

grads = module * direction * 255.0
plt.figure("Gradients")
plt.imshow(grads)


colormap = plt.get_cmap("gist_rainbow")
values = np.tile(np.arange(0, 360), (30, 1))/360.0
image = colormap(values)

#plt.figure("Colormap", figsize=(2, 1))
#plt.imshow(image)

plt.show()


#cv2.imshow('frame', frame)  # params search
#cv2.imshow('gaus', blur)  # params search

#cv2.imshow('median', median)

#cv2.imshow('laplacian', laplacian)  # params search

#cv2.imshow('sobelx', sobelx)  # magnitude
#cv2.imshow('sobely', sobely)  # vector



#cv2.waitKey()

#cv2.destroyAllWindows()
# opencv python example
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_gradients/py_gradients.html
