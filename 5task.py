import cv2 as cv
from shared import imgs

max_value = 255
max_type = 6
max_binary_value = 255
trackbar_type = 'Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'
trackbar_value = 'Value'
window_name = 'Thresholds'

trackbar_otsu = 'Otsu'

def threshold_demo(val):
    # 0: Binary
    # 1: Binary Inverted
    # 2: Threshold Truncated
    # 3: Threshold to Zero
    # 4: Threshold to Zero Inverted

    # 5: Adaptive ADAPTIVE_THRESH_MEAN_C
    # 6: ADAPTIVE_THRESH_GAUSSIAN_C
    threshold_type = cv.getTrackbarPos(trackbar_type, window_name)
    threshold_value = cv.getTrackbarPos(trackbar_value, window_name)
    threshold_otsu = cv.getTrackbarPos(trackbar_otsu, window_name)
    dst = None
    if threshold_type < 5:
        if threshold_otsu == 1:
            threshold_type = threshold_type + cv.THRESH_OTSU
        _, dst = cv.threshold(src_gray, threshold_value, max_binary_value, threshold_type)
    else:
        dst = cv.adaptiveThreshold(src_gray, 255, threshold_type - 5, cv.THRESH_BINARY, 11, 2)

    cv.imshow(window_name, dst)


src = cv.imread("/Users/kolsha/Pictures/LJqnAmEmPtg.jpg")
if src is None:
    print('Could not open or find the image')
    exit(0)


src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
cv.namedWindow(window_name)
cv.createTrackbar(trackbar_type, window_name, 3, max_type, threshold_demo)

cv.createTrackbar(trackbar_value, window_name, 0, max_value, threshold_demo)

cv.createTrackbar(trackbar_otsu, window_name, 0, 1, threshold_demo)

threshold_demo(0)

cv.waitKey()
