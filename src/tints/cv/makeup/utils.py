import cv2
import imutils
import dlib
import threading
from src.tints.settings import SHAPE_68_PATH, SHAPE_81_PATH
from src.tints.cv.simulation.apply_eyeshadow import Eyeshadow
from src.tints.cv.simulation.apply_blush import blush
from src.tints.cv.simulation.apply_lipstick import lipstick
from src.tints.cv.simulation.apply_concealer import concealer
from src.tints.cv.simulation.apply_foundation import foundation

class Color:
    def __init__(self, r=0, g=0, b=0, intensity=.7):
        self.r = r
        self.g = g
        self.b = b
        self.intensity = intensity

    def __str__(self):
        return str({'r': self.r, 'g': self.g, 'b': self.b, 'intensity': self.intensity})

    def values(self):
        return [self.r, self.g, self.b, self.intensity]


class Globals:
    landmarks = {}
    makeup_workers = {}
    camera_index = 0
    output_frame = None
    video_feed_enabled = False
    cap = cv2.VideoCapture()
    detector = dlib.get_frontal_face_detector()
    face_pose_predictor_68 = dlib.shape_predictor(SHAPE_68_PATH)
    face_pose_predictor_81 = dlib.shape_predictor(SHAPE_81_PATH)


def eyeshadow_worker(w_frame, r, g, b, intensity, out_queue) -> None:
    eyes = Eyeshadow()
    result = eyes.apply_eyeshadow(
        w_frame,
        Globals.landmarks['68_landmarks_x'], Globals.landmarks['68_landmarks_y'],
        r, g, b, intensity
    )
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    out_queue.append({
        'image': result,
        'range': (eyes.x_all, eyes.y_all)
    })


def blush_worker(w_frame, r, g, b, intensity, out_queue) -> None:
    cheeks = blush()
    result = cheeks.apply_blush(
        w_frame,
        Globals.landmarks['68_landmarks_x'], Globals.landmarks['68_landmarks_y'],
        r, g, b, intensity
    )
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    out_queue.append({
        'image': result,
        'range': (cheeks.x_all, cheeks.y_all)
    })


def lipstick_worker(w_frame, r, g, b, intensity, l_type, gloss, out_queue) -> None:
    lip = lipstick(w_frame)
    result = lip.apply_lipstick(
        Globals.landmarks['68_landmarks_x'], Globals.landmarks['68_landmarks_y'],
        r, g, b, l_type, gloss
    )
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    out_queue.append({
        'image': result,
        'range': (lip.x_all, lip.y_all)
    })


def concealer_worker(w_frame, r, g, b, intensity, k_h, k_w, out_queue) -> None:
    face_con = concealer()
    result = face_con.apply_blush(
        w_frame,
        Globals.landmarks['68_landmarks_x'], Globals.landmarks['68_landmarks_y'],
        r, g, b, k_h, k_w, intensity)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    out_queue.append({
        'image': result,
        'range': (face_con.x_all, face_con.y_all)
    })


def foundation_worker(w_frame, r, g, b, intensity, k_h, k_w, out_queue) -> None:
    face_foundation = foundation()
    result = face_foundation.apply_foundation(
        w_frame,
        Globals.landmarks['81_landmarks_x'], Globals.landmarks['81_landmarks_y'],
        Globals.landmarks['68_landmarks_x'], Globals.landmarks['68_landmarks_y'],
        r, g, b, k_h, k_w, intensity
    )
    
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    out_queue.append({
        'image': result,
        'range': (face_foundation.x_all, face_foundation.y_all)
    })


Globals.makeup_workers = {
    'eyeshadow_worker':     { 'function': eyeshadow_worker,     'args': [], 'enabled': False },
    'blush_worker':         { 'function': blush_worker,         'args': [], 'enabled': False },
    'lipstick_worker':      { 'function': lipstick_worker,      'args': [], 'enabled': False },
    'concealer_worker':     { 'function': concealer_worker,     'args': [], 'enabled': False },
    'foundation_worker':    { 'function': foundation_worker,    'args': [], 'enabled': False },
}


def join_makeup_workers(w_frame, w_landmarks_x, w_landmarks_y):
    threads = []
    shared_queue = []

    for makeup_worker in Globals.makeup_workers:
        worker = Globals.makeup_workers[makeup_worker]

        if worker['enabled']:
            t = threading.Thread(target=worker['function'],
                                 args=(w_frame, *worker['args'], shared_queue),
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
            pose_landmarks = Globals.face_pose_predictor_68(gray, face)
            for i in range(68):
                landmarks_x.append(pose_landmarks.part(i).x)
                landmarks_y.append(pose_landmarks.part(i).y)

            Globals.landmarks['68_landmarks_x'] = landmarks_x
            Globals.landmarks['68_landmarks_y'] = landmarks_y

            if Globals.makeup_workers['foundation_worker']['enabled']:
                pose_landmarks = Globals.face_pose_predictor_81(gray, face)
                for i in range(81):
                    landmarks_x.append(pose_landmarks.part(i).x)
                    landmarks_y.append(pose_landmarks.part(i).y)

                Globals.landmarks['81_landmarks_x'] = landmarks_x
                Globals.landmarks['81_landmarks_y'] = landmarks_y

            filter_res = join_makeup_workers(frame2, landmarks_x, landmarks_y)

            final_result = filter_res if filter_res is not None else Globals.output_frame

            # The following line is for testing with cv2 imshow
            # return final_result

            (flag, encodedImage) = cv2.imencode(".png", final_result)
            
            # ensure the frame was successfully encoded
            if not flag:
                continue
            # yield the output frame in the byte format
            yield (b'--frame\r\n' b'Content-Type: image/png\r\n\r\n' +
                bytearray(encodedImage) + b'\r\n')



def enable_makeup(makeup_state, r=0, g=0, b=0, intensity=.7, lipstick_type='hard', gloss=False, k_h=81, k_w=81):
    if makeup_state == 'eyeshadow':
        Globals.makeup_workers['eyeshadow_worker']['args'] = [*Color(r, g, b, intensity).values()]
        Globals.makeup_workers['eyeshadow_worker']['enabled'] = True
    elif makeup_state == 'lipstick':
        Globals.makeup_workers['lipstick_worker']['args'] = [*Color(r, g, b, intensity).values(), lipstick_type, gloss]
        Globals.makeup_workers['lipstick_worker']['enabled'] = True
    elif makeup_state == 'blush':
        Globals.makeup_workers['blush_worker']['args'] = [*Color(r, g, b, intensity).values()]
        Globals.makeup_workers['blush_worker']['enabled'] = True
    elif makeup_state == 'concealer':
        Globals.makeup_workers['concealer_worker']['args'] = [*Color(r, g, b, intensity).values(), k_h, k_w]
        Globals.makeup_workers['concealer_worker']['enabled'] = True
    elif makeup_state == 'foundation':
        Globals.makeup_workers['foundation_worker']['args'] = [*Color(r, g, b, intensity).values(), k_h, k_w]
        Globals.makeup_workers['foundation_worker']['enabled'] = True


def disable_makeup(makeup_state):
    if makeup_state == 'eyeshadow':
        Globals.makeup_workers['eyeshadow_worker']['enabled'] = False
    elif makeup_state == 'lipstick':
        Globals.makeup_workers['lipstick_worker']['enabled'] = False
    elif makeup_state == 'blush':
        Globals.makeup_workers['blush_worker']['enabled'] = False
    elif makeup_state == 'concealer':
        Globals.makeup_workers['concealer_worker']['enabled'] = False
    elif makeup_state == 'foundation':
        Globals.makeup_workers['foundation_worker']['enabled'] = False


def start_cam():
    Globals.cap.open(Globals.camera_index)
    Globals.video_feed_enabled = True


def stop_cam():
    Globals.video_feed_enabled = False
    Globals.cap.release()