import cv2
import imutils
import dlib
import time
from src.tints.settings import SHAPE_68_PATH
import threading
from src.tints.cv.simulation.apply_eyeshadow import eyeshadow
from src.tints.cv.simulation.apply_blush import blush


class Color:
    def __init__(self, r=0, g=0, b=0, intensity=.7):
        self.r = r
        self.g = g
        self.b = b
        self.intensity = intensity

    def __str__(self):
        return str({'r': self.r, 'g': self.g, 'b': self.b, 'intensity': self.intensity})


class MakeupGlobals:
    cap = None
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

    @classmethod
    def set_cap(cls, cap):
        cls.cap = cap


def eyeshadow_worker(w_frame, w_landmarks_x, w_landmarks_y, r, g, b, intensity, out_queue) -> None:
    # print("testing eyeshadow worker")
    eyes = eyeshadow(w_frame)
    result = eyes.apply_eyeshadow(w_landmarks_x, w_landmarks_y, r, g, b, intensity)
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


# MakeupGlobals.makeup_workers.append(eyeshadow_worker)
# MakeupGlobals.makeup_workers.append(blush_worker)
def toggle_makeup_worker(worker):
    if MakeupGlobals.makeup_workers[worker.__name__][2]:
        MakeupGlobals.makeup_workers[worker.__name__][2] = False
    else:
        MakeupGlobals.makeup_workers[worker.__name__][2] = True
    print(MakeupGlobals.makeup_workers[worker.__name__][1])


MakeupGlobals.makeup_workers = {
    'eyeshadow_worker': [eyeshadow_worker, MakeupGlobals.eye_color, False],
    'blush_worker': [blush_worker, MakeupGlobals.blush_color, False],
}


def join_makeup_workers(w_frame, w_landmarks_x, w_landmarks_y):
    threads = []
    shared_queue = []

    for makeup_worker in MakeupGlobals.makeup_workers:
        worker = MakeupGlobals.makeup_workers[makeup_worker]
        print(worker[2])
        if worker[2]:
            t = threading.Thread(target=worker[0],
                                 args=(w_frame, w_landmarks_x, w_landmarks_y, worker[1].r, worker[1].g, worker[1].b,
                                       worker[1].intensity, shared_queue),
                                 daemon=True)
            threads.append(t)
            t.start()

    if len(threads) > 0:
        for t in threads:
            t.join()

        final_image = shared_queue.pop()['image']

        while len(shared_queue) > 0:
            temp_img = shared_queue.pop()
            (range_x, range_y) = temp_img['range']
            temp_img = temp_img['image']

            for x, y in zip(range_x, range_y):
                final_image[x, y] = temp_img[x, y]

        return final_image

    else:
        return None


def apply_makeup_video():
    # global cap, outputFrame, eyeshadow_state, r_value, b_value, g_value
    prev = 0
    frame_rate = 15
    # _, frame = MakeupGlobals.cap.read()
    while True:
        ret, frame = MakeupGlobals.cap.read()
        MakeupGlobals.output_frame = frame
        time_elapsed = time.time() - prev
        frame = imutils.resize(frame, width=700)

        if time_elapsed > 1. / frame_rate:
            prev = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_faces = MakeupGlobals.detector(gray, 0)
            landmarks_x = []
            landmarks_y = []
            # try:
            for face in detected_faces:
                pose_landmarks = MakeupGlobals.face_pose_predictor(gray, face)
                for i in range(68):
                    landmarks_x.append(pose_landmarks.part(i).x)
                    landmarks_y.append(pose_landmarks.part(i).y)

                filter_res = join_makeup_workers(frame2, landmarks_x, landmarks_y)

                if filter_res is not None:
                    MakeupGlobals.output_frame = filter_res

                # Testing
                cv2.imshow("Frame", MakeupGlobals.output_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            # except Exception as e:
            #     print(e)


def handle_makeup_state(makeup_state, r=0, g=0, b=0, intensity=.7):
    # global eyeshadow_state, r_eye, g_eye, b_eye
    # global lipstick_state, r_lip, g_lip, b_lip
    if makeup_state == 'eyeshadow':
        MakeupGlobals.makeup_workers['eyeshadow_worker'][1] = Color(r, g, b, intensity)
        toggle_makeup_worker(eyeshadow_worker)
    # elif makeup_state == 'No_eyeshadow':
    #     rm_makeup_worker(eyeshadow_worker)
    if makeup_state == 'lipstick':
        # MakeupGlobals.lipstick_state = True
        MakeupGlobals.lip_color = Color(r, g, b, intensity)
    # elif makeup_state == 'No_lipstick':
    #     MakeupGlobals.lipstick_state = False
    if makeup_state == 'blush':
        MakeupGlobals.makeup_workers['blush_worker'][1] = Color(r, g, b, intensity)
        toggle_makeup_worker(blush_worker)
    # elif makeup_state == 'No_blush':
    #     MakeupGlobals.blush_state = False


def handle_makeup_video(frame, landmarks_x, landmarks_y):
    print("testing handle makeup")
    # global r_eye, g_eye, b_eye
    # if MakeupGlobals.eyeshadow_state:
    #     MakeupGlobals.makeup_workers.append((eyeshadow_worker, MakeupGlobals.eye_color))

    MakeupGlobals.output_frame = join_makeup_workers(frame, landmarks_x, landmarks_y)
    # if \
    # len(MakeupGlobals.makeup_workers) > 0 else frame


def opencam():
    # global cap, frame_rate, prev
    print("[INFO] camera sensor warming up...")
    MakeupGlobals.cap = cv2.VideoCapture(0)
    time.sleep(2.0)
    # MakeupGlobals.eyeshadow_state = False
    apply_makeup_video()


def caprelease():
    # global cap
    MakeupGlobals.cap.release()
    cv2.destroyAllWindows()


def generate():
    # grab global references 
    # global outputFrame
    # loop over frames from the output stream
    while True:
        # check if the output frame is available, otherwise skip
        # the iteration of the loop
        if MakeupGlobals.output_frame is None:
            continue
        # encode the frame in JPEG format
        (flag, encodedImage) = cv2.imencode(".jpg", MakeupGlobals.output_frame)
        # ensure the frame was successfully encoded
        if not flag:
            continue
        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')
