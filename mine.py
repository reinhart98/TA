import os
import numpy as np
import cv2
from PIL import Image
import pickle
import sys

#face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
face_cascade = cv2.CascadeClassifier('/home/odroid/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_alt2.xml')
#recognizer = cv2.createLBPHFaceRecognizer()
recognizer = cv2.face.createLBPHFaceRecognizer()
colec = cv2.face.MinDistancePredictCollector()
recognizer.load("trainer.yml")

labels = {"persons_name":1}
with open("labels.pickle", "rb") as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in labels.items()}
		
		
cap = cv2.VideoCapture(0)

while(True):
	#video cap
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
	
	for (x,y,w,h) in faces:
		#print(x,y,w,h)
		roi_gray = gray[y:y+h, x:x+w]
		roy_color = frame[y:y+h, x:x+w]
		
		#recognize how?
		
		#id_ , conf = recognizer.predict_collect(roi_gray) #some error, some say cuz its opencv 3.1.0 bug 
		#conf = 0.7											#solution : up opencv to 3.3 or just use MinDistancePredictCollector(...)
		#if conf>=45 and conf<=85:
			#print(id_)
			#print(labels[id_])
		#elif conf <45:
			#print("unknown")
						
		id_ = recognizer.predict(roi_gray, colec)
		conf = colec.getDist()
		label = colec.getLabel()
		if conf>=40 and conf<=85:
			print(id_)				#conf sdh benar problem now dia nd baca id_ nya
			print(labels[id_])	# alhasi krn dia baca id_ nya none atau tdk ada, it cant read the labels
		else:
			print("unknown")
			
		img_item = "my-img.png"
		cv2.imwrite(img_item, roi_gray)
		
		color = (255, 0, 0)
		stroke = 2
		end_coord_x = x+w
		end_coord_y = y+h
		cv2.rectangle(frame, (x,y), (end_coord_x, end_coord_y), color, stroke)
		
	cv2.imshow('frame',frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break
		
cap.release()
cv2.destroyAllWindows()