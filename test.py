import cv2
import src.tints.cv.makeup.utils as mutils

mutils.start_cam()
# mutils.enable_makeup('eyeshadow', 200, 10, 20, 1)
# mutils.enable_makeup('blush', 130, 197, 81, .6)
mutils.enable_makeup('lipstick', 255, 0, 0, gloss=True)
# mutils.enable_makeup('concealer', 200, 10, 20)
mutils.enable_makeup('foundation', r=68, g=59, b=49)
while True:
    cv2.imshow("Frame", mutils.apply_makeup_video())
    key = cv2.waitKey(1) & 0xFF
