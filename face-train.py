# Creating a face-detection model using haarcascade_frontalface classifier

import os
import cv2
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(__file__)
image_dir = os.path.join(BASE_DIR, 'images')

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt.xml')
features = []
targets = []

current_id = 0
target_ids = {}


for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('png') or file.endswith('jpg'):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path))
            # print(label, path, sep = ': ')

            if label not in target_ids:
                target_ids[label] = current_id
                current_id += 1
            id_ = target_ids[label]

            
            # open image using pillow and convert to grayscale 
            pil_image = Image.open(path).convert('L') 

            # Resize image
            #final_image = pil_image.resize((540,540), Image.ANTIALIAS)

            # convert pil image into numpy array
            image_array = np.array(pil_image, 'uint8')
            # print(image_array)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor = 1.5, minNeighbors = 5)

            for (x, y, w, h) in faces:
                region_of_interest = image_array[y:y+h, x:x+w]
                features.append(region_of_interest)
                targets.append(id_)

# print(target_ids)

# Saving the dictionary target_ids to a file
with open('target_ids.pkl', 'wb') as f:
    pickle.dump(target_ids, f)


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(features, np.array(targets))
recognizer.save('trainer.yml')