import os
import numpy as np
import cv2
from PIL import Image
from google.cloud import storage
from pytel import tg
import pickle
import sys
import json
import requests
import telegram
import time
import os

#credential_path = '/home/odroid/TA/service.json'
#os.system('export GOOGLE_APPLICATION_CREDENTIALS="/home/odroid/TA/service.json"')

'''TOKEN = "536159039:AAH0o_BLr0CHpSoFABByJCFNCZaGE43XAX4"
URL = "https://api.telegram.org/bot{}/".format(TOKEN)

bot = telegram.Bot(token='536159039:AAH0o_BLr0CHpSoFABByJCFNCZaGE43XAX4')'''

client = storage.Client()
bucket = client.get_bucket('deep-freehold-213203.appspot.com')
blob = bucket.get_blob('trainerv2.yml')
with open('trainerv2.yml', 'wb') as file_obj:
	blob.download_to_file(file_obj)	


face_cascade = cv2.CascadeClassifier('/home/odroid/opencv-3.4.0/data/haarcascades/haarcascade_frontalface_alt2.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
#colec = cv2.face.MinDistancePredictCollector()
recognizer.read("trainerv2.yml")

labels = {"persons_name":0}
with open("labels.pickle", "rb") as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}
		
		
cap = cv2.VideoCapture(0)

telegram = tg.Telegram('unix:///tmp/tg.sock') # For Unix Domain

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
		
		id_ , conf = recognizer.predict(roi_gray) #some error, some say cuz its opencv 3.1.0 bug 
																#solution : up opencv to 3.3 or just use MinDistancePredictCollector(...)
		if conf>=45 and conf<=80:
			#print(id_)
			#print(labels[id_])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (255,255,255)
			stroke = 2
			cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
			#telegram = tg.Telegram('unix:///tmp/tg.sock') # For Unix Domain Socket
			msg = name
			telegram.send_message('david_reinhart','ada tamu datang '+msg)
		elif conf > 80:
			unk = 'unknown'
			#print(unk)
			font = cv2.FONT_HERSHEY_SIMPLEX
			color = (255,255,255)
			stroke = 2
			cv2.putText(frame,unk,(x,y),font,1,color,stroke,cv2.LINE_AA)
			#telegram = tg.Telegram('unix:///tmp/tg.sock') # For Unix Domain Socket
			msg = 'identitas tak diketahui'
			telegram.send_message('david_reinhart','ada tamu datang '+msg)
			img_item = "unk.png"
			cv2.imwrite(img_item, roy_color)
			telegram.send_image('david_reinhart', '/home/odroid/TA/unk.png')
		
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
