import cv2
import src.tints.cv.makeup.utils as mutils

mutils.start_cam()
# mutils.enable_makeup('eyeshadow', 34, 74, 162, .7)
# mutils.enable_makeup('blush', 87, 36, 51, .5)
# mutils.enable_makeup('eyeliner', 142, 30, 29, 1)
mutils.enable_makeup('lipstick', 142, 30, 29, gloss=True, lipstick_type='soft')
# mutils.enable_makeup('concealer', 200, 10, 20, 1)
# mutils.enable_makeup('foundation', 255, 253, 208)
while True:
    cv2.imshow("Frame", mutils.apply_makeup_video())
    key = cv2.waitKey(1) & 0xFF
