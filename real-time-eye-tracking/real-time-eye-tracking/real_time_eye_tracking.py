import cv2
import numpy as np
###############################################################################
# Sources:
# https://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_face_detection.html#gsc.tab=0
# https://medium.com/@stepanfilonov/tracking-your-eyes-with-python-3952e66194a6
###############################################################################
face_cascade = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_eye.xml')
detector_params = cv2.SimpleBlobDetector_Params() # eye tracking using blob detection algorithm
detector_params.filterByArea = True               # Use area filtering for better results
detector_params.maxArea = 1500                    # pupil size in pixels will be within 1500
detector = cv2.SimpleBlobDetector_create(detector_params)

###############################################################################
# detect_eyes : Params(image of a face, eye_cascade)
# given an image of a face, find the right and left eye and return them
###############################################################################
def detect_eyes(img, classifier):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = classifier.detectMultiScale(gray_img, 1.3, 5) # get the eyes from the face
    width = img.shape[1]  # the width of the face
    height = img.shape[0] # the height of the face
    left_eye = None
    right_eye = None

    for (x,y,w,h) in eyes:
        if y+h > height/2: # pass if the eye is at the bottom
            pass 
        eye_center = x + w / 2 # Center of the eye
        if eye_center < width / 2:
            left_eye = img[y:y + h, x:x + w]
        else:
            right_eye = img[y:y + h, x:x + w]
    return left_eye, right_eye

###############################################################################
# detect_faces : Params(image of a person, face_cascade)
# given an image of a person, determine which is the face and return it
###############################################################################
def detect_faces(img, classifier):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray_img, 1.3, 5)
    if len(faces) > 1:
        biggest = (0,0,0,0)
        for i in faces:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(faces) == 1:
        biggest = faces
    else:
        return None
    for (x,y,w,h) in biggest:
        faces = img[y:y + h, x:x + w]
    return faces

###############################################################################
# cut_eyebrows : Params(image of an eye haar classifier)
# given an image of an eye and eyebrow, remove the eyebrow and return result
###############################################################################
def cut_eyebrows(img):
    height, width, _ = img.shape
    eyebrow_h = int(height /4)
    img = img[eyebrow_h:height, 0:width] # cut eyebrows out
    return img

###############################################################################
# blob_process : Params()
# detect and draw blobs on an image - we are passing in the eyes here
###############################################################################
def blob_process(img, threshold, detector):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2)
    img = cv2.dilate(img, None, iterations=4)
    img = cv2.medianBlur(img, 5)
    keypoints = detector.detect(img)
    return keypoints

def nothing(x):
    pass

###############################################################################
# Main
###############################################################################
def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('image')
    cv2.createTrackbar('threshold', 'image', 0, 255, nothing)
    while True:
        _, frame = cap.read()
        face_frame = detect_faces(frame, face_cascade)
        if face_frame is not None:
            eyes = detect_eyes(face_frame, eye_cascade)
            for eye in eyes:
                if eye is not None:
                    threshold = r = cv2.getTrackbarPos('threshold', 'image')
                    eye = cut_eyebrows(eye)
                    keypoints = blob_process(eye, threshold, detector)
                    eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()