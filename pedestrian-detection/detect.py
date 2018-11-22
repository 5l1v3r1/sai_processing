# USAGE
# python detect.py --images images

# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2


# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# loop over the image paths
cap = cv2.VideoCapture('/Users/kolsha/Documents/Projects/Python/sai_processing/videos/vtest.avi')
while(1):
        ret, frame = cap.read()
        if frame is None:
            break
        image = imutils.resize(frame, width=min(frame.shape[0], frame.shape[1]))
        orig = image.copy()

        # detect people in the image
        (rects, weights) = hog.detectMultiScale(image, winStride=(8,8), padding=(32,32), scale=1.005)

        # draw the original bounding boxes
        for (x, y, w, h) in rects:
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.165)

        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

        # show some information on the number of bounding boxes
        # filename = imagePath[imagePath.rfind("/") + 1:]
        # print("[INFO] {}: {} original boxes, {} after suppression".format(
        #     filename, len(rects), len(pick)))

        img = np.concatenate((orig, image), axis=1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow("Ped", img)
        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
        	break