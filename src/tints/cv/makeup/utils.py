import cv2
import imutils
import dlib
import time
import threading
from src.tints.settings import SHAPE_68_PATH, SHAPE_81_PATH
from src.tints.cv.simulation.apply_eyeshadow import Eyeshadow
from src.tints.cv.simulation.apply_blush import blush
from src.tints.cv.simulation.apply_lipstick import lipstick
from src.tints.cv.simulation.apply_concealer import Concealer
from src.tints.cv.simulation.apply_foundation import foundation
from .eyeliner import Eyeliner

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
    makeup_args = []
    camera_index = 0
    output_frame = None
    prev_time = 0
    frame_rate = 30
    padding = 50
    face_resized_width = 250
    video_feed_enabled = False
    cap = cv2.VideoCapture()
    detector = dlib.get_frontal_face_detector()
    face_pose_predictor_68 = dlib.shape_predictor(SHAPE_68_PATH)
    face_pose_predictor_81 = dlib.shape_predictor(SHAPE_81_PATH)


def foundation_worker(w_frame, r, g, b, intensity, k_h, k_w, out_queue) -> None:
    face_foundation = foundation()
    result = face_foundation.apply_foundation(
        w_frame,
        Globals.landmarks['81_landmarks_x'], Globals.landmarks['81_landmarks_y'],
        Globals.landmarks['68_landmarks_x'], Globals.landmarks['68_landmarks_y'],
        r, g, b, k_h, k_w, intensity
    )
    
    out_queue.append({
        'index': 0,
        'image': result,
        'range': (face_foundation.x_all, face_foundation.y_all)
    })


def concealer_worker(w_frame, r, g, b, intensity, k_h, k_w, out_queue) -> None:
    face_con = Concealer()
    result = face_con.apply_concealer(
        w_frame,
        Globals.landmarks['68_landmarks_x'], Globals.landmarks['68_landmarks_y'],
        r, g, b, k_h, k_w, intensity)

    out_queue.append({
        'index': 1,
        'image': result,
        'range': (face_con.x_all, face_con.y_all)
    })


def blush_worker(w_frame, r, g, b, intensity, out_queue) -> None:
    cheeks = blush()
    result = cheeks.apply_blush(
        w_frame,
        Globals.landmarks['68_landmarks_x'], Globals.landmarks['68_landmarks_y'],
        r, g, b, intensity
    )

    out_queue.append({
        'index': 2,
        'image': result,
        'range': (cheeks.x_all, cheeks.y_all)
    })


def eyeshadow_worker(w_frame, r, g, b, intensity, out_queue) -> None:
    eyes = Eyeshadow()
    result = eyes.apply_eyeshadow(
        w_frame,
        Globals.landmarks['68_landmarks_x'], Globals.landmarks['68_landmarks_y'],
        r, g, b, intensity
    )

    out_queue.append({
        'index': 3,
        'image': result,
        'range': (eyes.x_all, eyes.y_all)
    })


def eyeliner_worker(w_frame, r, g, b, intensity, out_queue) -> None:
    eye = Eyeliner(w_frame)
    result = eye.apply_eyeliner(
        Globals.landmarks['68_landmarks_x'], Globals.landmarks['68_landmarks_y'],
        r, g, b, intensity
    )

    out_queue.append({
        'index': 4,
        'image': result,
        'range': (eye.x_all, eye.y_all)
    })

def lipstick_worker(w_frame, r, g, b, intensity, l_type, gloss, out_queue) -> None:
    lip = lipstick(w_frame)
    result = lip.apply_lipstick(
        Globals.landmarks['68_landmarks_x'], Globals.landmarks['68_landmarks_y'],
        r, g, b, l_type, gloss
    )

    out_queue.append({
        'index': 5,
        'image': result,
        'range': (lip.x_all, lip.y_all)
    })


Globals.makeup_workers = {
    'foundation_worker':    { 'function': foundation_worker,    'args': [], 'enabled': False },
    'eyeshadow_worker':     { 'function': eyeshadow_worker,     'args': [], 'enabled': False },
    'eyeliner_worker':      { 'function': eyeliner_worker,      'args': [], 'enabled': False },
    'blush_worker':         { 'function': blush_worker,         'args': [], 'enabled': False },
    'lipstick_worker':      { 'function': lipstick_worker,      'args': [], 'enabled': False },
    'concealer_worker':     { 'function': concealer_worker,     'args': [], 'enabled': False },
}


def join_makeup_workers(w_frame):
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
        shared_queue = sorted(shared_queue, key=lambda x: x['index'], reverse=True)

        final_image = shared_queue.pop()['image']

        while len(shared_queue) > 0:
            temp_img = shared_queue.pop()
            (range_x, range_y) = temp_img['range']
            temp_img = temp_img['image']

            for x, y in zip(range_x, range_y):
                final_image[x, y] = temp_img[x, y]

        final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)

        return final_image

    return None


def apply_makeup_video():
    if not Globals.video_feed_enabled: return Globals.output_frame


    while True:
        time_elapsed = time.time() - Globals.prev_time

        if time_elapsed < 1./Globals.frame_rate:
            return Globals.output_frame

        Globals.prev_time = time.time()

        _, frame = Globals.cap.read()
        frame = imutils.resize(frame, width = 1000)
        Globals.output_frame = frame

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_faces = Globals.detector(gray, 0)
        landmarks_x_68 = []
        landmarks_y_68 = []
        landmarks_x_81 = []
        landmarks_y_81 = []

        for face in detected_faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            height, width = frame2.shape[:2]

            orignal_face_width = x2-x1
            ratio = Globals.face_resized_width / orignal_face_width
            new_padding = int(Globals.padding / ratio)
            new_y1= max(y1-new_padding,0)
            new_y2= min(y2+new_padding,height)
            new_x1= max(x1-new_padding,0)
            new_x2= min(x2+new_padding,width)
            cropped_img = frame2[ new_y1:new_y2, new_x1:new_x2]
            cropped_img = imutils.resize(cropped_img, width = (Globals.face_resized_width + 2 * Globals.padding))

            pose_landmarks = Globals.face_pose_predictor_68(gray, face)

            for i in range(68):
                landmarks_x_68.append(int(((pose_landmarks.part(i).x) - new_x1) * ratio))
                landmarks_y_68.append(int(((pose_landmarks.part(i).y) - new_y1) * ratio))

            Globals.landmarks['68_landmarks_x'] = landmarks_x_68
            Globals.landmarks['68_landmarks_y'] = landmarks_y_68

            if Globals.makeup_workers['foundation_worker']['enabled']:
                pose_landmarks = Globals.face_pose_predictor_81(gray, face)

                for i in range(81):
                    landmarks_x_81.append(int(((pose_landmarks.part(i).x) - x1) * ratio))
                    landmarks_y_81.append(int(((pose_landmarks.part(i).y) - y1) * ratio))

                Globals.landmarks['81_landmarks_x'] = landmarks_x_81
                Globals.landmarks['81_landmarks_y'] = landmarks_y_81

            filter_res = join_makeup_workers(cropped_img)

            if filter_res is not None:                
                filter_res = imutils.resize(filter_res, width=new_x2 - new_x1)
                cheight, cwidth = filter_res.shape[:2]
                frame[ new_y1:new_y1+cheight, new_x1:new_x1+cwidth] = filter_res

            # The following line is for testing with cv2 imshow
            # return frame

            (flag, encodedImage) = cv2.imencode(".png", frame)
            
            # ensure the frame was successfully encoded
            if not flag:
                continue
            # yield the output frame in the byte format
            yield (b'--frame\r\n' b'Content-Type: image/png\r\n\r\n' +
                bytearray(encodedImage) + b'\r\n')



def enable_makeup(makeup_type='', r=0, g=0, b=0, intensity=.7, lipstick_type='hard', gloss=False, k_h=81, k_w=81):
    if makeup_type == 'eyeshadow':
        Globals.makeup_workers['eyeshadow_worker']['args'] = [*Color(r, g, b, intensity).values()]
        Globals.makeup_workers['eyeshadow_worker']['enabled'] = True
    elif makeup_type == 'lipstick':
        Globals.makeup_workers['lipstick_worker']['args'] = [*Color(r, g, b, intensity).values(), lipstick_type, gloss]
        Globals.makeup_workers['lipstick_worker']['enabled'] = True
    elif makeup_type == 'blush':
        Globals.makeup_workers['blush_worker']['args'] = [*Color(r, g, b, intensity).values()]
        Globals.makeup_workers['blush_worker']['enabled'] = True
    elif makeup_type == 'concealer':
        Globals.makeup_workers['concealer_worker']['args'] = [*Color(r, g, b, intensity).values(), k_h, k_w]
        Globals.makeup_workers['concealer_worker']['enabled'] = True
    elif makeup_type == 'foundation':
        Globals.makeup_workers['foundation_worker']['args'] = [*Color(r, g, b, intensity).values(), k_h, k_w]
        Globals.makeup_workers['foundation_worker']['enabled'] = True
    elif makeup_type == 'eyeliner':
        Globals.makeup_workers['eyeliner_worker']['args'] = [*Color(r, g, b, intensity).values()]
        Globals.makeup_workers['eyeliner_worker']['enabled'] = True


Globals.makeup_args = enable_makeup.__code__.co_varnames


def disable_makeup(makeup_type):
    if makeup_type == 'eyeshadow':
        Globals.makeup_workers['eyeshadow_worker']['enabled'] = False
    elif makeup_type == 'lipstick':
        Globals.makeup_workers['lipstick_worker']['enabled'] = False
    elif makeup_type == 'blush':
        Globals.makeup_workers['blush_worker']['enabled'] = False
    elif makeup_type == 'concealer':
        Globals.makeup_workers['concealer_worker']['enabled'] = False
    elif makeup_type == 'foundation':
        Globals.makeup_workers['foundation_worker']['enabled'] = False
    elif makeup_type == 'eyeliner':
        Globals.makeup_workers['eyeliner_worker']['enabled'] = False


def start_cam():
    Globals.cap.open(Globals.camera_index)
    Globals.video_feed_enabled = True


def stop_cam():
    Globals.video_feed_enabled = False
    Globals.cap.release()