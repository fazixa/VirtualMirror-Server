# Import necessary libraries
import threading

from flask import Flask, render_template, Response
import cv2
import atexit
import src.tints.cv.makeup.utils as mutils
from multiprocessing import Process

# Initialize the Flask app
app = Flask(__name__, template_folder='templates')


# def gen_frames():
#     while True:
#         success, frame = camera.read()  # read the camera frame
#         if not success:
#             break
#         else:
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/open-cam', methods=['POST'])
def open_cam():
    # t = threading.Thread(target=mutils.start_cam, daemon=True)
    # t.start()
    return "Success opening cam", 200


@app.route('/close-cam', methods=['POST'])
def close_cam():
    mutils.stop_cam()
    return 'Cam closed'


@app.route('/video-feed')
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    print("video feed")
    return Response(mutils.apply_makeup_video(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/blush')
def blush():
    mutils.handle_makeup_state('blush', 130, 197, 81, .6)
    return 'ok'


@app.route('/eyeshadow')
def eyeshadow():
    mutils.handle_makeup_state('eyeshadow', 130, 197, 81, .6)
    return 'ok'


if __name__ == '__main__':
    mutils.MakeupGlobals.cap = cv2.VideoCapture(0)
    app.run(debug=True)
