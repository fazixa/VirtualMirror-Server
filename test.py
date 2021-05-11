# from src.tints.cv.makeup.utils import Color
# import time

# def spread(a, b, c):
#     res = a + b + c

# a = [1, 2, 3]

# # t = time.time()
# print(time.time())
# for _ in range(100000):
#     spread(*a)
# print(time.time())

# # print(time.time() - t)

# # t = time.time()
# print(time.time())

# for _ in range(100000):
#     spread(a[0], a[1], a[2])
# print(time.time())

# print(time.time() - t)
import cv2
import src.tints.cv.makeup.utils as mutils

mutils.start_cam()
# mutils.enable_makeup('blush', 130, 197, 81, .6)
mutils.enable_makeup('eyeshadow', 200, 10, 20, 1)
mutils.enable_makeup('lipstick', 200, 10, 20, gloss=True)
mutils.enable_makeup('concealer', 200, 10, 20)
# mutils.enable_makeup('foundation', 200, 10, 20)
while True:
    cv2.imshow("Frame", mutils.apply_makeup_video())
    key = cv2.waitKey(1) & 0xFF