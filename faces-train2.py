import os
import numpy as np
import cv2
from PIL import Image
import pickle
from google.cloud import storage


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('/home/odroid/opencv-3.4.0/data/haarcascades/haarcascade_frontalface_alt2.xml')
#recognizer = cv2.LBPHFaceRecognizer()
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file)
			label = os.path.basename(root).replace(" ", "-").lower()
			print(label, path)

			if not label in label_ids:
				label_ids[label] = current_id
				current_id+=1
			id_ = label_ids[label]
			print(label_ids)


			#x_train.append(path)
			#y_labels.append(label)

			pil_images = Image.open(path).convert("L")
			size = (300,300)
			final_image = pil_images.resize(size, Image.ANTIALIAS)
			image_array = np.array(final_image, "uint8")
			#print(image_array)

			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

			for(x,y,w,h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_labels.append(id_)

#print(x_train)
#print(y_labels)

with open("labels.pickle", "wb") as f:
	pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")

client = storage.Client()
bucket = client.get_bucket('deep-freehold-213203.appspot.com')
blob = bucket.blob('trainerv2.yml') # membuat file .yml
blob.upload_from_filename('/home/odroid/TA/trainer.yml') # mengupload file .yml dengan contentx



