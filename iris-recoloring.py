from fdlite import FaceDetection, FaceLandmark, face_detection_to_roi
from fdlite import IrisLandmark, iris_roi_from_face_landmarks
from fdlite.examples import iris_recoloring
from PIL import Image
import cv2
import numpy as np

EXCITING_NEW_EYE_COLOR = (74, 136, 237)

# load detection models
detect_faces = FaceDetection()
detect_face_landmarks = FaceLandmark()
detect_iris = IrisLandmark()

# open image
# img = Image.open('./face.jpg')
# detect face

cap = cv2.VideoCapture(0)

while True:

	_, img = cap.read()
	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	img = Image.fromarray(img)

	face_detections = detect_faces(img)
	if len(face_detections) > 0:
		# get ROI for the first face found
		face_roi = face_detection_to_roi(face_detections[0], img.size)
		# detect face landmarks
		face_landmarks = detect_face_landmarks(img, face_roi)
		# get ROI for both eyes
		eye_roi = iris_roi_from_face_landmarks(face_landmarks, img.size)
		left_eye_roi, right_eye_roi = eye_roi
		# detect iris landmarks for both eyes
		left_eye_results = detect_iris(img, left_eye_roi)
		right_eye_results = detect_iris(img, right_eye_roi, is_right_eye=True)
		# change the iris color
		iris_recoloring.recolor_iris(img, left_eye_results, iris_color=EXCITING_NEW_EYE_COLOR)
		iris_recoloring.recolor_iris(img, right_eye_results, iris_color=EXCITING_NEW_EYE_COLOR)
		# img.show()
		cv2.imshow("Frame", np.array(img)[:, :, ::-1])
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		print('no face detected :(')