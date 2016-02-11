# import the necessary packages
from __future__ import print_function
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import argparse
import imutils
import os
import numpy as np
import cv2
from PIL import Image
from google.cloud import storage
#from pytel import tg
import pickle
import sys
import json
import requests
import telegram
import time
import os

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=100,
	help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=-1,
	help="Whether or not frames should be displayed")
args = vars(ap.parse_args())

TOKEN = "536159039:AAH0o_BLr0CHpSoFABByJCFNCZaGE43XAX4"
URL = "https://api.telegram.org/bot{}/".format(TOKEN)

bot = telegram.Bot(token='536159039:AAH0o_BLr0CHpSoFABByJCFNCZaGE43XAX4')
chatid = 482880664

face_cascade = cv2.CascadeClassifier('/home/odroid/opencv-3.4.0/data/haarcascades/haarcascade_frontalface_alt2.xml')
#face_cascade = cv2.CascadeClassifier('/home/odroid/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_alt2.xml')
#recognizer = cv2.createLBPHFaceRecognizer()
recognizer = cv2.face.LBPHFaceRecognizer_create()
#colec = cv2.face.MinDistancePredictCollector()
recognizer.read("trainer.yml")

labels = {"persons_name":0}
with open("labels.pickle", "rb") as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}
		
		
stream = cv2.VideoCapture(0)
print("[INFO] sampling frames from webcam...")
fps = FPS().start()

while fps._numFrames < args["num_frames"]:
	#video cap
	(grabbed, frame) = stream.read()
	frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
	
	for (x,y,w,h) in faces:
		#print(x,y,w,h)
		roi_gray = gray[y:y+h, x:x+w]
		roy_color = frame[y:y+h, x:x+w]
		
		#recognize how?
		
		id_ , conf = recognizer.predict(roi_gray) #some error, some say cuz its opencv 3.1.0 bug 
																#solution : up opencv to 3.3 or just use MinDistancePredictCollector(...)
		if conf>=45 and conf<=85:
			#print(id_)
			#print(labels[id_])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (255,255,255)
			stroke = 2
			cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
			msg = name
			#time.sleep(2)
			bot.send_message(chatid , text='ada tamu '+msg)
			time.sleep(.100)
		elif conf > 85:
			unk = 'unknown'
			#print(unk)
			font = cv2.FONT_HERSHEY_SIMPLEX
			color = (255,255,255)
			stroke = 2
			cv2.putText(frame,unk,(x,y),font,1,color,stroke,cv2.LINE_AA)
			#telegram = tg.Telegram('unix:///tmp/tg.sock') # For Unix Domain Socket
			msg = 'identitas tak diketahui'
			#time.sleep(2)
			bot.send_message(chatid, text='Ada tamu '+msg)
			time.sleep(.200)

			
	#	img_item = "my-img.png"
	#	cv2.imwrite(img_item, roy_color)
		
		color = (255, 0, 0)
		stroke = 2
		end_coord_x = x+w
		end_coord_y = y+h
		cv2.rectangle(frame, (x,y), (end_coord_x, end_coord_y), color, stroke)
		
	if args["display"] > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
 
	# update the FPS counter
	fps.update()
 
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
print("num of frame : {:.2f}".format(fps.numframe()))
		
stream.release()
cv2.destroyAllWindows()


print("[INFO] sampling THREADED frames from webcam...")
vs = WebcamVideoStream(src=0).start()
fps = FPS().start()

while fps._numFrames < args["num_frames"]:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
	
	for (x,y,w,h) in faces:
		#print(x,y,w,h)
		roi_gray = gray[y:y+h, x:x+w]
		roy_color = frame[y:y+h, x:x+w]
		
		#recognize how?
		
		id_ , conf = recognizer.predict(roi_gray) #some error, some say cuz its opencv 3.1.0 bug 
																#solution : up opencv to 3.3 or just use MinDistancePredictCollector(...)
		if conf>=45 and conf<=85:
			#print(id_)
			#print(labels[id_])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (255,255,255)
			stroke = 2
			cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
			msg = name
			#time.sleep(2)
			bot.send_message(chatid , text='ada tamu '+msg)
			time.sleep(.100)
		elif conf > 85:
			unk = 'unknown'
			#print(unk)
			font = cv2.FONT_HERSHEY_SIMPLEX
			color = (255,255,255)
			stroke = 2
			cv2.putText(frame,unk,(x,y),font,1,color,stroke,cv2.LINE_AA)
			#telegram = tg.Telegram('unix:///tmp/tg.sock') # For Unix Domain Socket
			msg = 'identitas tak diketahui'
			#time.sleep(2)
			bot.send_message(chatid, text='Ada tamu '+msg)
			time.sleep(.200)

			
	#	img_item = "my-img.png"
	#	cv2.imwrite(img_item, roy_color)
		
		color = (255, 0, 0)
		stroke = 2
		end_coord_x = x+w
		end_coord_y = y+h
		cv2.rectangle(frame, (x,y), (end_coord_x, end_coord_y), color, stroke)
	
 
	# check to see if the frame should be displayed to our screen
	if args["display"] > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
 
	# update the FPS counter
	fps.update()
 
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
print("num of frame : {:.2f}".format(fps.numframe()))
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
