from src.tints.cv.makeup.utils import opencam, handle_makeup_state
import time
import threading


if __name__ == '__main__':
    t = threading.Thread(target=opencam, daemon=True)
    t.start()

    time.sleep(5)

    handle_makeup_state('blush', 130, 197, 81, .6)
    time.sleep(5)

    handle_makeup_state('eyeshadow', 237,29,36, .3)
    time.sleep(5)

    handle_makeup_state('blush')
    time.sleep(5)

    handle_makeup_state('eyeshadow')
    time.sleep(5)

    # threading.Thread(target=handle_makeup_state, args=('No_eyeshadow', 10, 10, 100)).start()


# a = {}
#
#
# def test():
#     print('Hi')
#
#
# if __name__ == '__main__':
#     print(len(a))