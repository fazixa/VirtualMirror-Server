import cv2
import imutils
import dlib
import time
import threading
from src.tints.settings import SHAPE_68_PATH, SHAPE_81_PATH
from src.tints.cv.simulation.apply_foundation import Foundation
from src.tints.cv.simulation.apply_concealer import Concealer
from src.tints.cv.simulation.apply_blush import Blush
from src.tints.cv.simulation.apply_eyeshadow import Eyeshadow
from src.tints.cv.simulation.apply_eyeliner import Eyeliner
from src.tints.cv.simulation.apply_lipstick import Lipstick
import traceback

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
    prev_time = 0
    output_frame = None
    frame_rate = 30
    padding = 50
    face_resized_width = 250
    video_feed_enabled = False
    # Motion Detection Vars
    prev_frame = None
    motion_detected = False
    #######################
    foundation = Foundation()
    concealer = Concealer()
    blush = Blush()
    eyeshadow = Eyeshadow()
    eyeliner = Eyeliner()
    lipstick = Lipstick()
    cap = cv2.VideoCapture()
    detector = dlib.get_frontal_face_detector()
    face_pose_predictor_68 = dlib.shape_predictor(SHAPE_68_PATH)
    face_pose_predictor_81 = dlib.shape_predictor(SHAPE_81_PATH)


def foundation_worker(w_frame, r, g, b, intensity, k_h, k_w, out_queue) -> None:
    result = Globals.foundation.apply_foundation(
        w_frame,
        Globals.landmarks['81_landmarks_x'], Globals.landmarks['81_landmarks_y'],
        Globals.landmarks['68_landmarks_x'], Globals.landmarks['68_landmarks_y'],
        r, g, b, k_h, k_w, intensity
    )
    
    out_queue.append({
        'index': 0,
        'image': result,
        'range': (Globals.foundation.x_all, Globals.foundation.y_all)
    })


def concealer_worker(w_frame, r, g, b, intensity, k_h, k_w, out_queue) -> None:
    result = Globals.concealer.apply_concealer(
        w_frame,
        Globals.landmarks['68_landmarks_x'], Globals.landmarks['68_landmarks_y'],
        r, g, b, k_h, k_w, intensity)

    out_queue.append({
        'index': 1,
        'image': result,
        'range': (Globals.concealer.x_all, Globals.concealer.y_all)
    })


def blush_worker(w_frame, r, g, b, intensity, out_queue) -> None:
    result = Globals.blush.apply_blush(
        w_frame,
        Globals.landmarks['68_landmarks_x'], Globals.landmarks['68_landmarks_y'],
        r, g, b, intensity
    )

    out_queue.append({
        'index': 2,
        'image': result,
        'range': (Globals.blush.x_all, Globals.blush.y_all)
    })


def eyeshadow_worker(w_frame, r, g, b, intensity, out_queue) -> None:
    result = Globals.eyeshadow.apply_eyeshadow(
        w_frame,
        Globals.landmarks['68_landmarks_x'], Globals.landmarks['68_landmarks_y'],
        r, g, b, 0.85
    )

    out_queue.append({
        'index': 3,
        'image': result,
        'range': (Globals.eyeshadow.x_all, Globals.eyeshadow.y_all)
    })


def eyeliner_worker(w_frame, r, g, b, intensity, out_queue) -> None:
    result = Globals.eyeliner.apply_eyeliner(
        w_frame, 
        Globals.landmarks['68_landmarks_x'], Globals.landmarks['68_landmarks_y'],
        r, g, b, 1
    )

    out_queue.append({
        'index': 4,
        'image': result,
        'range': (Globals.eyeliner.x_all, Globals.eyeliner.y_all)
    })

def lipstick_worker(w_frame, r, g, b, intensity, l_type, gloss, out_queue) -> None:
    result = Globals.lipstick.apply_lipstick(
        w_frame,
        Globals.landmarks['68_landmarks_x'], Globals.landmarks['68_landmarks_y'],
        r, g, b, intensity, l_type, gloss
    )

    out_queue.append({
        'index': 5,
        'image': result,
        'range': (Globals.lipstick.x_all, Globals.lipstick.y_all)
    })


Globals.makeup_workers = {
    'lipstick_worker':      { 'function': lipstick_worker,      'instance': Globals.lipstick,   'args': [], 'enabled': False },
    'eyeliner_worker':      { 'function': eyeliner_worker,      'instance': Globals.eyeliner,   'args': [], 'enabled': False },
    'eyeshadow_worker':     { 'function': eyeshadow_worker,     'instance': Globals.eyeshadow,  'args': [], 'enabled': False },
    'blush_worker':         { 'function': blush_worker,         'instance': Globals.blush,      'args': [], 'enabled': False },
    'concealer_worker':     { 'function': concealer_worker,     'instance': Globals.concealer,  'args': [], 'enabled': False },
    'foundation_worker':    { 'function': foundation_worker,    'instance': Globals.foundation, 'args': [], 'enabled': False },
}


def join_makeup_workers(w_frame):
    threads = []
    shared_list = []

    for makeup_worker in Globals.makeup_workers:
        worker = Globals.makeup_workers[makeup_worker]

        if worker['enabled']:
            t = threading.Thread(
                target=worker['function'],
                args=(w_frame, *worker['args'], shared_list),
                daemon=True
            )
            threads.append(t)

    if len(threads) > 0:
        for t in threads:
            t.start()
            t.join()

    if len(shared_list) > 0:
        shared_list = sorted(shared_list, key=lambda x: x['index'], reverse=True)

        final_image = shared_list.pop()['image']

        while len(shared_list) > 0:
            temp_img = shared_list.pop()
            (range_x, range_y), temp_img = temp_img['range'], temp_img['image']

            # for x, y in zip(range_x, range_y):
            final_image[range_x, range_y] = temp_img[range_x, range_y]

        final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)

        return final_image

    return w_frame


def join_makeup_workers_static(w_frame):
    shared_list = []

    for makeup_worker in Globals.makeup_workers:
        worker = Globals.makeup_workers[makeup_worker]

        if worker['enabled']:
            shared_list.append({
                'image': worker['instance'].im_copy,
                'range': (worker['instance'].x_all, worker['instance'].y_all)
            })
    
    while len(shared_list) > 0:
        temp_img = shared_list.pop()
        (range_x, range_y), temp_img = temp_img['range'], temp_img['image']

        for x, y in zip(range_x, range_y):
            w_frame[x, y] = temp_img[x, y]

    w_frame = cv2.cvtColor(w_frame, cv2.COLOR_RGB2BGR)

    return w_frame


def apply_makeup_video():
    # Return last recorded image of None if does not exist
    if not Globals.video_feed_enabled: return Globals.output_frame

    while True:
        time_elapsed = time.time() - Globals.prev_time

        if time_elapsed < 1./Globals.frame_rate:
            return Globals.output_frame

        Globals.prev_time = time.time()

        _, frame = Globals.cap.read()

        # frame = imutils.resize(frame, width = 1000)

        Globals.output_frame = frame

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Motion Detection
        if Globals.prev_frame is None:
            Globals.prev_frame = gray
            continue
    
        frame_diff = cv2.absdiff(Globals.prev_frame, gray)

        frame_thresh = cv2.threshold(frame_diff, 2, 255, cv2.THRESH_BINARY)[1] 
        frame_thresh = cv2.dilate(frame_thresh, None, iterations=2) 

        cnts, _ = cv2.findContours(frame_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

        for contour in cnts: 
            temp = cv2.contourArea(contour)
            if temp < 150000: 
                continue
            # print(temp)
            Globals.motion_detected = True

        try:
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

                # =====================================================
                '''
                Cropping face with padding to cover forehead and chin
                '''
                orignal_face_width = x2-x1
                ratio = Globals.face_resized_width / orignal_face_width
                new_padding = int(Globals.padding / ratio)
                new_y1= max(y1-new_padding,0)
                new_y2= min(y2+new_padding,height)
                new_x1= max(x1-new_padding,0)
                new_x2= min(x2+new_padding,width)
                cropped_img = frame2[ new_y1:new_y2, new_x1:new_x2]
                cropped_img = imutils.resize(cropped_img, width = (Globals.face_resized_width + 2 * Globals.padding))
                # ======================================================

                if Globals.motion_detected:
                    print('motion detected')
                    
                    pose_landmarks = Globals.face_pose_predictor_68(gray, face)

                    for i in range(68):
                        landmarks_x_68.append(int(((pose_landmarks.part(i).x) - new_x1) * ratio))
                        landmarks_y_68.append(int(((pose_landmarks.part(i).y) - new_y1) * ratio))

                    Globals.landmarks['68_landmarks_x'] = landmarks_x_68
                    Globals.landmarks['68_landmarks_y'] = landmarks_y_68

                    if Globals.makeup_workers['foundation_worker']['enabled']:
                        pose_landmarks = Globals.face_pose_predictor_81(gray, face)

                        for i in range(81):
                            landmarks_x_81.append(int(((pose_landmarks.part(i).x) - new_x1) * ratio))
                            landmarks_y_81.append(int(((pose_landmarks.part(i).y) - new_y1) * ratio))

                        Globals.landmarks['81_landmarks_x'] = landmarks_x_81
                        Globals.landmarks['81_landmarks_y'] = landmarks_y_81

                    filter_res = join_makeup_workers(cropped_img)

                    Globals.motion_detected = False

                else:
                    print('no motion detected')
                    filter_res = join_makeup_workers_static(cropped_img)
            
                filter_res = imutils.resize(filter_res, width=new_x2 - new_x1)
                cheight, cwidth = filter_res.shape[:2]
                frame[ new_y1:new_y1+cheight, new_x1:new_x1+cwidth] = filter_res

        except Exception as e:
            traceback.print_exc()

        Globals.prev_frame = gray.copy()

        # The following line is for testing with cv2 imshow
        return frame

        # (flag, encodedImage) = cv2.imencode(".png", frame)
        
        # # ensure the frame was successfully encoded
        # if not flag:
        #     continue
        # # yield the output frame in the byte format
        # yield (b'--frame\r\n' b'Content-Type: image/png\r\n\r\n' +
        #     bytearray(encodedImage) + b'\r\n')



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