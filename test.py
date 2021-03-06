import cv2
# import src.cv.makeup.utils as mutils

# mutils.start_cam()
# mutils.enable_makeup('eyeshadow', 34, 74, 162, .7)
# mutils.enable_makeup('blush', 87, 36, 51, .5)
# mutils.enable_makeup('eyeliner', 142, 30, 29, .5)
# mutils.enable_makeup('lipstick', 142, 30, 29, gloss=False, lipstick_type='soft')
# # mutils.enable_makeup('concealer', 200, 10, 20, 1)
# # mutils.enable_makeup('foundation', 255, 253, 208, .3)
# while True:
#     cv2.imshow("Frame", mutils.apply_makeup_video())
#     key = cv2.waitKey(1) & 0xFF

import matplotlib.pyplot as plt
import numpy as np
from src.cv.simulation.apply_foundation import Foundation
from src.settings import SHAPE_68_PATH, SHAPE_81_PATH
import dlib

detector = dlib.get_frontal_face_detector()
face_pose_predictor_68 = dlib.shape_predictor(SHAPE_68_PATH)
face_pose_predictor_81 = dlib.shape_predictor(SHAPE_81_PATH)

cap = cv2.VideoCapture(0)

while True:
    x_68 = []
    y_68 = []
    x_81 = []
    y_81 = []

    # img = cv2.imread('face.jpg')
    _, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face = detector(gray, 0)[0]

    pose_landmarks = face_pose_predictor_68(gray, face)

    for i in range(68):
        x_68.append(int(((pose_landmarks.part(i).x))))
        y_68.append(int(((pose_landmarks.part(i).y))))


    a = np.min(x_68[36:41])
    b = np.max(x_68[36:41])
    c = np.min(y_68[36:41])
    d = np.max(y_68[36:41])

    img = img[c:d, a:b]

    # ret,thresh1 = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)

    # pose_landmarks = face_pose_predictor_81(gray, face)
    # for i in range(81):
    #     x_81.append(int(((pose_landmarks.part(i).x))))
    #     y_81.append(int(((pose_landmarks.part(i).y))))

    # y_68 = np.array(y_68)

    # y_68 += 10

    # foundation = Foundation()

    # res = foundation.apply_foundation(frame, x_81, y_81, x_68, y_68, 255, 253, 208, 81, 81, 1)

    # cv2.imshow('frame', cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # xs, ys = np.r_[x_68[1:17], x_81[68:81]], np.r_[y_68[1:17], y_81[68:81]]

    # for x, y in zip(x_68, y_68):
    #     img = cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

    # img = img[np.c_[x_68[36:42], y_68[36:42]]]


    cv2.imshow('frame', img)
    key = cv2.waitKey(1) & 0xFF
    # cv2.waitKey(0)
    # ax1.scatter(y_68[1:17], y_81[68:81])