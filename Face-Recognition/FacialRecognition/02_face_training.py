import cv2
import numpy as np
from PIL import Image
import os
import re

# Path for face image database
path = 'att_faces'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# function to get the images and label data
def getImagesAndLabels(path):

    dirPaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    for dir in dirPaths:
        if not os.path.isdir(dir):
            continue
        imagePaths = [os.path.join(dir,f) for f in os.listdir(dir)]  
        result = re.findall(r'\d{1,2}', imagePaths[0])
        id = int(result[0])
        for imagePath in imagePaths:
            if not imagePath.endswith(".pgm") and not imagePath.endswith(".jpg"):
                continue
            PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
            img_numpy = np.array(PIL_img,'uint8')

            faces = detector.detectMultiScale(img_numpy)

            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)

    return faceSamples, ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)

print(faces,ids)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.save('trainer.yml') # recognizer.save() worked on Mac, but not on Pi

# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
