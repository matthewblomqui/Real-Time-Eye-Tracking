import cv2
import numpy as np
###############################################################################
# Sources:
# https://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_face_detection.html#gsc.tab=0
# https://medium.com/@stepanfilonov/tracking-your-eyes-with-python-3952e66194a6
###############################################################################

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

img = cv2.imread('stock_img.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert image to gray scale
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#for (x,y,w,h) in faces:
#    cv2.rectangle(img, (x,y), (x+w,y+h),(255,255,0),2)



cv2.imshow('frame', gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()