"""
 * @author Faezeh Shayesteh
 * @email shayesteh.fa@gmail.com
 * @create date 2021-04-23 04:02:39
 * @modify date 2021-04-23 16:05:09
 * @desc [description]
"""

from __future__ import division
import cv2
from pylab import *
import numpy as np
from numpy import c_
from skimage import color
from scipy.interpolate import interp1d


class Eyeliner(object):

    def __init__(self, img):
        self.r = 0
        self.g = 0
        self.b = 0
        self.intensity = 0
        self.eyeshadow_height = 2
        # Original image
        self.image = img
        # All the changes will be applied to im_copy
        self.im_copy = self.image.copy()
        self.height, self.width = self.image.shape[:2]
        self.x_all = []
        self.y_all = []

    # -------- general methods ----------
    def interpolate(self, lx=[], ly=[], k1='quadratic'):
        unew = np.arange(lx[0], lx[-1] + 1, 1)
        f2 = interp1d(lx, ly, kind=k1)
        return (f2, unew)

    def apply_color(self, x, y):
        # converting desired parts of the original image to LAB color space
        lip_LAB = color.rgb2lab((self.im_copy[x, y] / 255.).reshape(len(x), 1, 3)).reshape(len(x), 3)
        # calculating mean of each channel
        L, A, B = mean(lip_LAB[:, 0]), mean(lip_LAB[:, 1]), mean(lip_LAB[:, 2])
        # converting the color of the makeup to LAB
        L1, A1, B1 = color.rgb2lab(np.array((self.r / 255., self.g / 255., self.b / 255.)).reshape(1, 1, 3)).reshape(
            3, )
        # applying the makeup color on image
        G = L1 / L
        lip_LAB = lip_LAB.reshape(len(x), 1, 3)
        lip_LAB[:, :, 1:3] = self.intensity * np.array([A1, B1]) + (1 - self.intensity) * lip_LAB[:, :, 1:3]
        lip_LAB[:, :, 0] = lip_LAB[:, :, 0] * (1 + self.intensity * (G - 1))
        # converting back toRGB
        self.im_copy[x, y] = color.lab2rgb(lip_LAB).reshape(len(x), 3) * 255

    def apply_blur(self, x, y):
        # gussian blur
        filter = np.zeros((self.height, self.width))
        cv2.fillConvexPoly(filter, np.array(c_[y, x], dtype='int32'), 1)
        filter = cv2.GaussianBlur(filter, (31, 31), 0)
        # Erosion to reduce blur size
        kernel = np.ones((7, 7), np.uint8)
        filter = cv2.erode(filter, kernel, iterations=1)
        alpha = np.zeros([self.height, self.width, 3], dtype='float64')
        alpha[:, :, 0] = filter
        alpha[:, :, 1] = filter
        alpha[:, :, 2] = filter
        self.im_copy = (alpha * self.im_copy + (1 - alpha) * self.image).astype('uint8')

    def fill(self, lower_path, upper_path):
        x = []
        y = []
        for i in lower_path[1]:
            for j in range(int(upper_path[0](i)), int(lower_path[0](i))):
                x.append(int(j))
                y.append(int(i))
        return x, y

    # -------- non general methods ----------
    def get_point(self, x, y):

        # left eye
        x_left_eye_lower = np.r_[x[17], x[36], x[37],x[38],x[39]]
        x_left_eye_upper = x_left_eye_lower
        y_left_eye_lower = np.r_[y[17]+25, y[36], y[37],y[38],y[39]]
        y_left_eye_upper = np.r_[y[17]+20, y[36]-15, y[37]-11,y[38]-7,y[39]-5]
        print(y_left_eye_lower)
        print(y_left_eye_upper)

        # right eye
        x_right_eye_lower = np.r_[x[42:46], x[26]-2]
        x_rigth_eye_upper = x_right_eye_lower 
        y_right_eye_lower = np.r_[y[42],y[43],y[44],y[45], y[26]+25]
        y_right_eye_upper = np.r_[y[42]-5,y[43]-7,y[44]-11,y[45]-15, y[26]+20]

        return x_left_eye_lower, x_left_eye_upper, y_left_eye_lower, y_left_eye_upper, \
               x_right_eye_lower, x_rigth_eye_upper, y_right_eye_lower, y_right_eye_upper

    def apply_eyeliner(self, landmarks_x, landmarks_y, r, g, b, intensity):

        self.r = r
        self.g = g
        self.b = b
        self.intensity = intensity
        # get eyes point
        x_left_eye_lower, x_left_eye_upper, y_left_eye_lower, y_left_eye_upper, x_right_eye_lower, \
        x_rigth_eye_upper, y_right_eye_lower, y_right_eye_upper = self.get_point(landmarks_x, landmarks_y)
        # create interpolated path
        left_eye_lower_path = self.interpolate(x_left_eye_lower[:], y_left_eye_lower[:], 'quadratic')
        left_eye_upper_path = self.interpolate(x_left_eye_upper[:], y_left_eye_upper[:], 'quadratic')
        right_eye_lower_path = self.interpolate(x_right_eye_lower[:], y_right_eye_lower[:], 'quadratic')
        right_eye_upper_path = self.interpolate(x_rigth_eye_upper[:], y_right_eye_upper[:], 'quadratic')
        # filling points eyelids
        x_all_left, y_all_left = self.fill(left_eye_lower_path, left_eye_upper_path)
        x_all_right, y_all_right = self.fill(right_eye_lower_path, right_eye_upper_path)
        x_all = x_all_left + x_all_right
        y_all = y_all_left + y_all_right
        self.x_all = x_all
        self.y_all = y_all

        # apply color
        self.apply_color(x_all, y_all)
        self.apply_blur(x_all, y_all)
        return self.im_copy