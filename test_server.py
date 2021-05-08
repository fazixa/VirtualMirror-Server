# Import necessary libraries
import threading

from flask import Flask, render_template, Response
import cv2
import atexit
import src.tints.cv.makeup.utils as mutils
from multiprocessing import Process

# Initialize the Flask app
app = Flask(__name__, template_folder='templates')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/open-cam', methods=['POST'])
def open_cam():
    mutils.start_cam()
    return "Success opening cam", 200


@app.route('/close-cam', methods=['POST'])
def close_cam():
    mutils.stop_cam()
    return 'Cam closed'


@app.route('/video-feed')
def video_feed():
    return Response(mutils.apply_makeup_video(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/blush')
def blush():
    mutils.enable_makeup('blush', 130, 197, 81, .6)
    return 'ok'


@app.route('/eyeshadow')
def eyeshadow():
    mutils.enable_makeup('eyeshadow', 130, 197, 81, .6)
    return 'ok'


if __name__ == '__main__':
    mutils.Globals.cap = cv2.VideoCapture(0)
    app.run(debug=True)
