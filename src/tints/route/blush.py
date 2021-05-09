import io
from skimage import io as skimage_io
import os
from flask import Flask, request, send_from_directory, jsonify, Blueprint
from flask_cors import cross_origin
from PIL import Image
from base64 import encodebytes
from src.tints.utils.json_encode import JSONEncoder
from src.tints.cv.simulation.apply_blush import blush
from src.tints.settings import SIMULATOR_INPUT, SIMULATOR_OUTPUT
import cv2
import time
import imutils
from flask import Flask, render_template, url_for, request, Response
import dlib
from src.tints.settings import SHAPE_68_PATH

blushr = Blueprint('blushr', __name__)

detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(SHAPE_68_PATH)
# This method executes before any API request


@blushr.before_request
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
@blushr.route('/api/makeup/image/blush', methods=['POST'])
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

    landmarks_x = []
    landmarks_y = []
    for i in range(68):
        landmarks_x.append(pose_landmarks.part(i).x)
        landmarks_y.append(pose_landmarks.part(i).y)

    r_value = request.form.get('r_value')
    g_value = request.form.get('g_value')
    b_value = request.form.get('b_value')

    blush_makeup = blush()
    
    img = blush_makeup.apply_blush(
        user_image,landmarks_x, landmarks_y, r_value, g_value, b_value, 1.5)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predict_result_intense = save_iamge(img,r_value,g_value,b_value,"blush",1.5)

    img = blush_makeup.apply_blush(
        user_image,landmarks_x, landmarks_y, r_value, g_value, b_value, 0.4)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predict_result_medium = save_iamge(img,r_value,g_value,b_value,"blush",0.4)

    img = blush_makeup.apply_blush(
        user_image,landmarks_x, landmarks_y, r_value, g_value, b_value,0.1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predict_result_fade = save_iamge(img,r_value,g_value,b_value,"blush",0.1)

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