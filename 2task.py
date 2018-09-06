import numpy as np
import cv2



# Load an color image in grayscale
image = cv2.imread('/Users/kolsha/Pictures/LJqnAmEmPtg.jpg')
#print img.size#[0,0].sum()/3

output = image.copy()


cv2.circle(output, (480, 390), 80, (0, 255, 0), 4)
cv2.rectangle(output, (430, 350), (530, 440), (0, 128, 255), -1)

cv2.imshow("output", np.hstack([ image, output]))
cv2.waitKey(0)