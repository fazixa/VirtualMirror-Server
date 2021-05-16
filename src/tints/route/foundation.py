import io
from skimage import io as skimage_io
import os
from flask import Flask, request, send_from_directory, jsonify, Blueprint
from flask_cors import cross_origin
from PIL import Image
from base64 import encodebytes
from src.tints.utils.json_encode import JSONEncoder
from src.tints.cv.simulation.apply_eyeshadow import Eyeshadow
from src.tints.cv.simulation.apply_foundation import foundation
from src.tints.settings import SIMULATOR_INPUT, SIMULATOR_OUTPUT
import cv2
import time
import imutils
from flask import Flask, render_template, url_for, request, Response
import dlib
from src.tints.settings import SHAPE_68_PATH,SHAPE_81_PATH

foundationm = Blueprint('foundationm', __name__)

detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(SHAPE_68_PATH)
face_pose_predictor_81 = dlib.shape_predictor(SHAPE_81_PATH)
# This method executes before any API request


@foundationm.before_request
def before_request():
    print('Start eyeshadow API request')

# --------------------------------- generl defs

def get_response_image(image_path):
    pil_img = Image.open(image_path, mode='r')  # reads the PIL image
    byte_arr = io.BytesIO()
    # convert the PIL image to byte array
    pil_img.save(byte_arr, format='JPEG')
    encoded_img = encodebytes(byte_arr.getvalue()).decode(
        'ascii')  # encode as base64
    return encoded_img

# ------------------------------------ non general
@foundationm.route('/api/makeup/image/foundation', methods=['POST'])
@cross_origin()
def simulator_lip():
    # check if the post request has the file part
    if 'user_image' not in request.files:
        return {"detail": "No file found"}, 400
    user_image = request.files['user_image']
    if user_image.filename == '':
        return {"detail": "Invalid file or filename missing"}, 400
    user_id = request.form.get('user_id')
    image_copy_name = 'simulated_image-{}.jpg'.format(str(user_id))
    user_image.save(os.path.join(SIMULATOR_INPUT, image_copy_name))
    user_image = skimage_io.imread(os.path.join(SIMULATOR_INPUT, image_copy_name))

    detected_faces = detector(user_image, 0)
    pose_landmarks = face_pose_predictor(user_image, detected_faces[0])
    pose_landmarks81 = face_pose_predictor_81(user_image, detected_faces[0])
    landmarks_x = []
    landmarks_y = []


    for face in detected_faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

    cropped_img = user_image[ y1:y2, x1:x2]
    height, width = user_image.shape[:2]
    cropped_width = x2-x1
    cropped_height = y2-y1
    ratio = 250 / cropped_width
    new_width = width * ratio
    cropped_img = imutils.resize(cropped_img, width=250)

    landmarks_x81 = []
    landmarks_y81 = []
    for i in range(68):
        landmarks_x.append(int(((pose_landmarks.part(i).x) - x1) * ratio))
        landmarks_y.append(int(((pose_landmarks.part(i).y) - y1) * ratio))
    for i in range(81):
        landmarks_x81.append(int(((pose_landmarks81.part(i).x) - x1) * ratio))
        landmarks_y81.append(int(((pose_landmarks81.part(i).y) - y1) * ratio))

    r_value = request.form.get('r_value')
    g_value = request.form.get('g_value')
    b_value = request.form.get('b_value')

    foundation_makeup = foundation()
    
    img = foundation_makeup.apply_foundation(
        cropped_img,landmarks_x, landmarks_y, landmarks_x81, landmarks_y81, r_value, g_value, b_value, 81, 81, 1.2)
    
    img = imutils.resize(img, width=cropped_width)
    cheight, cwidth = img.shape[:2]
    user_image_copy = user_image.copy()
    user_image_copy[ y1:y1+cheight, x1:x2] = img
    user_image_copy = cv2.cvtColor(user_image_copy, cv2.COLOR_BGR2RGB)
    predict_result_intense = save_iamge(user_image_copy,r_value,g_value,b_value,"foundation", 0.1)

    img = foundation_makeup.apply_foundation(
        cropped_img,landmarks_x, landmarks_y,landmarks_x81, landmarks_y81, r_value, g_value, b_value,81, 81, 0.3)
    img = imutils.resize(img, width=cropped_width)
    cheight, cwidth = img.shape[:2]
    user_image_copy = user_image.copy()
    user_image_copy[ y1:y1+cheight, x1:x2] = img
    user_image_copy = cv2.cvtColor(user_image_copy, cv2.COLOR_BGR2RGB)
    predict_result_medium = save_iamge(user_image_copy,r_value,g_value,b_value,"foundation",0.2)

    img = foundation_makeup.apply_foundation(
        cropped_img,landmarks_x, landmarks_y,landmarks_x81, landmarks_y81, r_value, g_value, b_value,81, 81,0.5)
    img = imutils.resize(img, width=cropped_width)
    cheight, cwidth = img.shape[:2]
    user_image_copy = user_image.copy()
    user_image_copy[ y1:y1+cheight, x1:x2] = img
    user_image_copy = cv2.cvtColor(user_image_copy, cv2.COLOR_BGR2RGB)
    predict_result_fade = save_iamge(user_image_copy,r_value,g_value,b_value,"foundation",0.3)

    result = [predict_result_intense,
              predict_result_medium, predict_result_fade]
    encoded_img = []
    for image_path in result:
        encoded_img.append(get_response_image(
            '{}/{}'.format(SIMULATOR_OUTPUT, image_path)))

    return (JSONEncoder().encode(encoded_img), 200)


def save_iamge(img,r_value,g_value,b_value,makeup_type,intensity):
    name = 'color_' + str(r_value) + '_' + \
    str(g_value) + '_' + str(b_value)
    file_name = '{}_output-{}_{}.jpg'.format(makeup_type,intensity, name)
    cv2.imwrite(os.path.join(SIMULATOR_OUTPUT, file_name), img)
    return file_name