import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
people = ['Prabhjot','Sidak','Ishaan']

# features = np.load('features.npy')
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

# img = cv.imread(r'D:\openCV\photos\valid4.jpeg')
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def facedet(imga,gray):

    faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

    for(x,y,w,h) in faces_rect:
        faces_roi = gray[y:y+h, x:x+h]
        label, confidence = face_recognizer.predict(faces_roi)
        print(people[label]," with a confidence of ", confidence)
        print(str(people[label]))
        cv.putText(imga, str(people[label]), (x+w+10,y+w+20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0),2)
        cv.rectangle(imga, (x,y), (x+w,y+h), (0,255,0), 2)
        cv.imshow('detected face', imga)
    # cv.waitKey(0)

capture = cv.VideoCapture(0)
while True:
    isTrue, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('video',frame)
    facedet(frame,gray)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()

