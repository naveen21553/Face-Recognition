# Program to use trained model to recognize and tag faces in video

import cv2
import pickle

vc = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
labels = {}
with open('target_ids.pkl', 'rb') as f:
    labels = pickle.load(f)
    targets =  {value: key.replace('-', ' ') for key, value in labels.items()}

while True:
    ret, frame = vc.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
    for (x, y, w, h) in faces:
        #print(x, y, w, h)
        region_of_interest = gray[x:x+w, y:y+h]
        id_, conf = recognizer.predict(region_of_interest)
        if conf >= 60:
            print(targets[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = targets[id_]
            color = (0,255,0)
            stroke = 1
            name = name + ', %.2f'%conf  
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,255), 2 )
        
    cv2.imshow('frame', frame)

    key = cv2.waitKey(20)
    if key == 27:
        break

vc.release()
cv2.destroyAllWindows()
