# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 11:42:58 2020

@author: Taha
"""
import face_recognition
from imutils import paths
import pickle 
import cv2
import os
import imutils
import time

imagePaths=list(paths.list_images('Images'))
knownEncodings=[]
knownNames=[]
for (i,imagePath) in enumerate(imagePaths):
    name=imagePath.split(os.path.sep)[-2]
    image=cv2.imread(imagePath)
    rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    boxes=face_recognition.face_locations(rgb,model='hog')    
    encodings=face_recognition.face_encodings(rgb,boxes)
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)
    data={'encodings':knownEncodings,'names':knownNames}
    f=open("face_enc","wb")
    f.write(pickle.dumps(data))
    f.close()


cascPathface = "haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPathface)
data = pickle.loads(open('face_enc', "rb").read())
print("Streaming started")
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    raise IOError("Cannot open webcam")

while True:

    ret,frame=video_capture.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(60,60),flags=cv2.CASCADE_SCALE_IMAGE)
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    encodings=face_recognition.face_encodings(rgb)
    names=[]
    for encoding in encodings:
        matches=face_recognition.compare_faces(data['encodings'],encoding)
        name='Unknown'
        if True in matches:
            matchedIdxs=[i for (i,b) in enumerate(matches) if b] 
            counts={}
            for i in matchedIdxs:
                name=data["names"][i]
                counts[name]=counts.get(name,0)+1
            name=max(counts,key=counts.get)
        names.append(name)
        # loop over the recognized faces
        for ((x, y, w, h), name) in zip(faces, names):
            # rescale the face coordinates
            # draw the predicted face name on the image
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()











