import cv2
import imutils
import dlib
import time
from src.tints.settings import SHAPE_68_PATH
import threading
from src.tints.cv.simulation.apply_eyeshadow import Eyeshadow
from src.tints.cv.simulation.apply_blush import blush


class Color:
    def __init__(self, r=0, g=0, b=0, intensity=.7):
        self.r = r
        self.g = g
        self.b = b
        self.intensity = intensity

    def __str__(self):
        return str({'r': self.r, 'g': self.g, 'b': self.b, 'intensity': self.intensity})


class Globals:
    def __init__(self):
        pass

    cap = cv2.VideoCapture()
    # r_value = None
    # g_value = None
    # b_value = None
    makeup_workers = {}
    output_frame = None
    lip_color = Color()
    eye_color = Color()
    blush_color = Color()
    blush_state = False
    lipstick_state = False
    eyeshadow_state = False
    detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(SHAPE_68_PATH)
    video_feed_enabled = True

    @classmethod
    def set_cap(cls, cap):
        cls.cap = cap


def eyeshadow_worker(w_frame, w_landmarks_x, w_landmarks_y, r, g, b, intensity, out_queue) -> None:
    eyes = Eyeshadow()
    result = eyes.apply_eyeshadow(w_frame, w_landmarks_x, w_landmarks_y, r, g, b, intensity)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    out_queue.append({
        'image': result,
        'range': (eyes.x_all, eyes.y_all)
    })


def blush_worker(w_frame, w_landmarks_x, w_landmarks_y, r, g, b, intensity, out_queue) -> None:
    cheeks = blush(w_frame)
    result = cheeks.apply_blush(w_landmarks_x, w_landmarks_y, r, g, b, intensity)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    out_queue.append({
        'image': result,
        'range': (cheeks.x_all, cheeks.y_all)
    })


Globals.makeup_workers = {
    'eyeshadow_worker': [eyeshadow_worker, Globals.eye_color, False],
    'blush_worker': [blush_worker, Globals.blush_color, False],
}


def join_makeup_workers(w_frame, w_landmarks_x, w_landmarks_y):
    threads = []
    shared_queue = []

    for makeup_worker in Globals.makeup_workers:
        worker = Globals.makeup_workers[makeup_worker]

        if worker[2]:
            t = threading.Thread(target=worker[0],
                                 args=(w_frame, w_landmarks_x, w_landmarks_y, worker[1].r, worker[1].g, worker[1].b,
                                       worker[1].intensity, shared_queue),
                                 daemon=True)
            threads.append(t)

    if len(threads) > 0:
        for t in threads:
            t.start()
            t.join()

    if len(shared_queue) > 0:

        final_image = shared_queue.pop()['image']

        while len(shared_queue) > 0:
            temp_img = shared_queue.pop()
            (range_x, range_y) = temp_img['range']
            temp_img = temp_img['image']

            for x, y in zip(range_x, range_y):
                final_image[x, y] = temp_img[x, y]

        return final_image

    return None


def apply_makeup_video():
    if not Globals.video_feed_enabled: return Globals.output_frame

    while True:
        _, frame = Globals.cap.read()
        Globals.output_frame = frame
        frame = imutils.resize(frame, width=700)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_faces = Globals.detector(gray, 0)
        landmarks_x = []
        landmarks_y = []

        for face in detected_faces:
            pose_landmarks = Globals.face_pose_predictor(gray, face)
            for i in range(68):
                landmarks_x.append(pose_landmarks.part(i).x)
                landmarks_y.append(pose_landmarks.part(i).y)

            filter_res = join_makeup_workers(frame2, landmarks_x, landmarks_y)

            if filter_res is not None:

                (flag, encodedImage) = cv2.imencode(".png", filter_res)
                # ensure the frame was successfully encoded
                if not flag:
                    continue
                # yield the output frame in the byte format
                yield (b'--frame\r\n' b'Content-Type: image/png\r\n\r\n' +
                    bytearray(encodedImage) + b'\r\n')
            
            else:
                (flag, encodedImage) = cv2.imencode(".png", Globals.output_frame)
                # ensure the frame was successfully encoded
                if not flag:
                    continue
                # yield the output frame in the byte format
                yield (b'--frame\r\n' b'Content-Type: image/png\r\n\r\n' +
                    bytearray(encodedImage) + b'\r\n')


# def handle_makeup_state(makeup_state, r=0, g=0, b=0, intensity=.7):
#     # global eyeshadow_state, r_eye, g_eye, b_eye
#     # global lipstick_state, r_lip, g_lip, b_lip
#     if makeup_state == 'eyeshadow':
#         Globals.makeup_workers['eyeshadow_worker'][1] = Color(r, g, b, intensity)
#         toggle_makeup_worker(eyeshadow_worker)
#     # elif makeup_state == 'No_eyeshadow':
#     #     rm_makeup_worker(eyeshadow_worker)
#     if makeup_state == 'lipstick':
#         # Globals.lipstick_state = True
#         Globals.lip_color = Color(r, g, b, intensity)
#     # elif makeup_state == 'No_lipstick':
#     #     Globals.lipstick_state = False
#     if makeup_state == 'blush':
#         Globals.makeup_workers['blush_worker'][1] = Color(r, g, b, intensity)
#         toggle_makeup_worker(blush_worker)
#     # elif makeup_state == 'No_blush':
#     #     Globals.blush_state = False


def enable_makeup(makeup_state, r=0, g=0, b=0, intensity=.7):
    if makeup_state == 'eyeshadow':
        Globals.makeup_workers['eyeshadow_worker'][1] = Color(r, g, b, intensity)
        Globals.makeup_workers['eyeshadow_worker'][2] = True
    elif makeup_state == 'lipstick':
        Globals.makeup_workers['lipstick_worker'][1] = Color(r, g, b, intensity)
        Globals.makeup_workers['lipstick_worker'][2] = True
    elif makeup_state == 'blush':
        Globals.makeup_workers['blush_worker'][1] = Color(r, g, b, intensity)
        Globals.makeup_workers['blush_worker'][2] = True


# def handle_makeup_video(frame, landmarks_x, landmarks_y):
#     print("testing handle makeup")
#     # global r_eye, g_eye, b_eye
#     # if Globals.eyeshadow_state:
#     #     Globals.makeup_workers.append((eyeshadow_worker, Globals.eye_color))

#     Globals.output_frame = join_makeup_workers(frame, landmarks_x, landmarks_y)
#     # if \
#     # len(Globals.makeup_workers) > 0 else frame


# def opencam():
#     # global cap, frame_rate, prev
#     print("[INFO] camera sensor warming up...")
#     Globals.cap = cv2.VideoCapture(0)
#     # time.sleep(2.0)
#     # Globals.eyeshadow_state = False
#     apply_makeup_video()
#     # threading.Thread(target=apply_makeup_video, daemon=True).start()


# def caprelease():
#     # global cap
#     Globals.cap.release()
#     cv2.destroyAllWindows()



def start_cam():
    Globals.cap.open(1)
    Globals.video_feed_enabled = True


def stop_cam():
    Globals.video_feed_enabled = False
    Globals.cap.release()


# def generate():
#     # grab global references 
#     # global outputFrame
#     # loop over frames from the output stream
#     #  Testing
#     # if key == ord("q"):
#     #     break

#     while True:
#         # check if the output frame is available, otherwise skip
#         # the iteration of the loop
#         if Globals.output_frame is None:
#             continue

#         # cv2.imshow("Frame", Globals.output_frame)
#         # key = cv2.waitKey(1) & 0xFF
#         # encode the frame in JPEG format
#         (flag, encodedImage) = cv2.imencode(".png", Globals.output_frame)
#         # ensure the frame was successfully encoded
#         if not flag:
#             continue
#         # yield the output frame in the byte format
#         yield (b'--frame\r\n' b'Content-Type: image/png\r\n\r\n' +
#                bytearray(encodedImage) + b'\r\n')
