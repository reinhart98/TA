import os
import numpy as np
import cv2
from PIL import Image
import pickle
import sys

faceDetect = cv2.CascadeClassifier('/home/odroid/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_alt2.xml')
eyeDetect = cv2.CascadeClassifier('/home/odroid/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_alt2.xml')

cam = cv2.VideoCapture(0)
rec = cv2.face.createLBPHFaceRecognizer()
rec.load('trainer.yml')
i=0
#font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX,3,1,0,3,1)
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.2, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        i=rec.predict(gray[y:y+h,x:x+w])
        conf = 0.7
        if (conf<50):
            if (i == 1):
                i="MS"
            elif (i==2):
                i="KR"
            else:
                i="Unknown" #had to put this to prevent the sample numbers such as 45 and 30 popping up
        else:
            i="Unknown"

        print(i)