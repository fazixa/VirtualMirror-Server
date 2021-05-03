import cv2
import imutils
import dlib
import time
from src.tints.settings import SHAPE_68_PATH
import threading
from src.tints.cv.simulation.apply_eyeshadow import Eyeshadow

detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(SHAPE_68_PATH)
outputFrame = None
eyeshadow_state = False
lipstick_state = False


def apply_makeup_video():
    
    global cap, outputFrame, eyeshadow_state, r_value,b_value,g_value
    prev = 0
    frame_rate = 15

    while True:
        ret, frame = cap.read()
        time_elapsed = time.time() - prev
        frame = imutils.resize(frame, width=700)

        if(time_elapsed > 1./frame_rate):
            prev = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_faces = detector(gray, 0)
            landmarks_x = []
            landmarks_y = []
            try:
                for face in detected_faces:
                    pose_landmarks = face_pose_predictor(gray, face)
                    for i in range(68):
                        landmarks_x.append(pose_landmarks.part(i).x)
                        landmarks_y.append(pose_landmarks.part(i).y)

                    frame = handle_makeup_Video(frame2,landmarks_x, landmarks_y)
                    outputFrame = frame
            except Exception as e:
                print(e)

def handle_makeup_on(makeup_state, r, g, b):
    global eyeshadow_state, r_eye, g_eye, b_eye
    global lipstick_state, r_lip , g_lip, b_lip
    if(makeup_state == 'eyeshadow'):
        eyeshadow_state = True
        r_eye = r
        g_eye = g
        b_eye = b
    if(makeup_state == 'lipstick'):
        lipstick_state = True
        r_lip = r
        g_lip = g
        b_lip = b

def handle_makeup_off(makeup_state):
    global eyeshadow_state
    if(makeup_state == 'eyeshadow'):
        print("offfffffffff")
        eyeshadow_state = False

def handle_makeup_Video(frame,landmarks_x,landmarks_y):
    global r_eye,g_eye,b_eye
    if (eyeshadow_state):
        eye = Eyeshadow()
        # print(r_eye,type(r_eye))
        frame = eye.apply_eyeshadow(frame,landmarks_x,landmarks_y,r_eye,g_eye,b_eye,0.7)
    return frame


def opencam():
    global cap, frame_rate, prev
    print("[INFO] camera sensor warming up...")
    cap = cv2.VideoCapture(0)
    time.sleep(2.0)
    print("time sleep finished")
    t = threading.Thread(target=apply_makeup_video, daemon=True)
    t.start()

def caprelease():
    global cap,eyeshadow_state
    eyeshadow_state = False
    cap.release()
    cv2.destroyAllWindows()

def generate():
    # grab global references 
    global outputFrame
    # loop over frames from the output stream
    while True:
        # check if the output frame is available, otherwise skip
        # the iteration of the loop
        if outputFrame is None:
            continue
        # encode the frame in JPEG format
        (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
        # ensure the frame was successfully encoded
        if not flag:
            continue
        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')