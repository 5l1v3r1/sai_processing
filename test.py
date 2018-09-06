import numpy as np
import cv2

# Load an color image in grayscale
img = cv2.imread('/Users/kolsha/Pictures/LJqnAmEmPtg.jpg')
#print img.size#[0,0].sum()/3

treshold = 127.0
count_of_px = 0
all_count = img.shape[0]*img.shape[1]

print all_count

for x in range(img.shape[0]):
	for y in range(img.shape[1]):
		bright = img[x,y].sum() / 3.0
		if bright > treshold:
			count_of_px += 1


print count_of_px / float(all_count)