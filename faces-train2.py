import os
import numpy as np
import cv2
from PIL import Image
import pickle


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('/home/odroid/opencv-3.4.0/data/haarcascades/haarcascade_frontalface_alt2.xml')
#recognizer = cv2.LBPHFaceRecognizer()
recognizer = cv2.face.LBPHFaceRecognizer_create()
warp_mode = cv2.MOTION_TRANSLATION

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
			sz = pil_images.shape
			
			if warp_mode == cv2.MOTION_HOMOGRAPHY:
				warp_matrix = np.eye(3, 3, dtype==np.float32)
			else:
				warp_MATRIX = np.eye(2, 3, dtype=np.float32)
				
			number_of_iteration = 5000
			term_eps = 1e-10
			criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iteration, term_eps)
			(cc, warp_matrix) = cv2.fimdTransformECC(pil_images, warp_matrix, criteria)
			if warp_mode == cv2.MOTION_HOMOGRAPHY:
				im_align = cv2.warpPerspective(pil_images, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
			else:
				im_align = cv2.warpAffine(pil_images, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

			
			#size = (300,300)
			#final_image = pil_images.resize(size, Image.ANTIALIAS)
			image_array = np.array(im_align, "uint8")
			#print(image_array)

			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
			
			if warp_mode == cv2.MOTION_HOMOGRAPHY:
				warp_matrix = np.eye(3, 3, dtype==np.float32)
			else:
				warp_MATRIX = np.eye(2, 3, dtype=np.float32)
				
			number_of_iteration = 5000
			term_eps = 1e-10
			criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iteration, term_eps)
			(cc, warp_matrix) = cv2.fimdTransformECC(pil_images, warp_matrix, criteria)
			if warp_mode == cv2.MOTION_HOMOGRAPHY:
				im_align = cv2.warpPerspective(pil_images, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
			else:
				im_align = cv2.warpAffine(pil_images, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

				
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



