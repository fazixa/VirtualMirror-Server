from __future__ import division
from itertools import zip_longest
# import scipy.interpolate
from scipy.interpolate import interp1d
import cv2
import numpy as np
from skimage import color
from PIL import Image
from imutils import face_utils
import os
import dlib
from pylab import *
from skimage import io

from scipy import interpolate


class concealer(object): 


    def apply_blush(self, img, landmark_x, landmark_y, r_value, g_value, b_value, ksize_h, ksize_w, intensity):
        self.red_b = int(r_value)
        self.green_b = int(g_value)
        self.blue_b = int(b_value)
        self.image = img
        # self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        # gray_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        # shape = self.get_cheek_shape(gray_image)
        self.image = Image.fromarray(self.image)
        self.image = np.asarray(self.image)
        self.height, self.width = self.image.shape[:2]
        self.image_copy = self.image.copy()

        indices_left = [1, 2, 3, 4, 48, 31, 36]
        # indices_face_bottom = range(1, 27)
        left_cheek_x = [landmark_x[i] for i in indices_left]
        face_bottom_y = [landmark_y[i] for i in indices_left]

        left_cheek_x, face_bottom_y = self.get_boundary_points(
            left_cheek_x, face_bottom_y)

        face_bottom_y, left_cheek_x = self.get_interior_points(
            left_cheek_x, face_bottom_y)

        self.__fill_blush_color(intensity)
        self.__smoothen_blush(left_cheek_x, face_bottom_y, ksize_h, ksize_w)

        indices_right = [15, 14, 13, 12, 54, 35, 45]
        face_top_x = [landmark_x[i] for i in indices_right]
        face_top_y = [landmark_y[i] for i in indices_right]
        face_top_x, face_top_y = self.get_boundary_points(
            face_top_x, face_top_y)
        face_top_y, face_top_x = self.get_interior_points(
            face_top_x, face_top_y)
        self.__fill_blush_color(intensity)
        self.__smoothen_blush(face_top_x, face_top_y, ksize_h, ksize_w)

        self.x_all = face_top_y
        self.y_all = face_top_x


        # # file_name = 'lip_output-' + name + '.jpg'

        # cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        return self.image_copy
        
    def get_boundary_points(self, x, y):
        tck, u = interpolate.splprep([x, y], s=0, per=1)
        unew = np.linspace(u.min(), u.max(), 1000)
        xnew, ynew = interpolate.splev(unew, tck, der=0)
        tup = c_[xnew.astype(int), ynew.astype(int)].tolist()
        coord = list(set(tuple(map(tuple, tup))))
        coord = np.array([list(elem) for elem in coord])
        return np.array(coord[:, 0], dtype=np.int32), np.array(coord[:, 1], dtype=np.int32)

    def get_interior_points(self, x, y):
        intx = []
        inty = []
        print('start get_interior_points')

        def ext(a, b, i):
            a, b = round(a), round(b)
            intx.extend(arange(a, b, 1).tolist())
            inty.extend((ones(b - a) * i).tolist())

        x, y = np.array(x), np.array(y)
        print('x,y get_interior_points')
        xmin, xmax = amin(x), amax(x)
        xrang = np.arange(xmin, xmax + 1, 1)
        print(type(xrang))
        print('x-rang')
        print(xrang)
        for i in xrang:
            try:
                ylist = y[where(x == i)]
                ext(amin(ylist), amax(ylist), i)
            except ValueError:  # raised if `y` is empty.
                pass

        print('xrang2 get_interior_points')
        return np.array(intx, dtype=np.int32), np.array(inty, dtype=np.int32)

    def __fill_blush_color(self, intensity):
        val = color.rgb2lab((self.image / 255.)
                            ).reshape(self.width * self.height, 3)
        L, A, B = np.mean(val[:, 0]), np.mean(val[:, 1]), np.mean(val[:, 2])
        L1, A1, B1 = color.rgb2lab(
            np.array((self.red_b / 255., self.green_b / 255., self.blue_b / 255.)).reshape(1, 1, 3)).reshape(3, )
        ll, aa, bb = (L1 - L) * intensity, (A1 - A) * \
            intensity, (B1 - B) * intensity
        val[:, 0] = np.clip(val[:, 0] + ll, 0, 100)
        val[:, 1] = np.clip(val[:, 1] + aa, -127, 128)
        val[:, 2] = np.clip(val[:, 2] + bb, -127, 128)
        self.image = color.lab2rgb(
            val.reshape(self.height, self.width, 3)) * 255
        # self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    def __smoothen_blush(self, x, y, ksize_h, ksize_w):
        # imgBase = np.zeros((self.height, self.height))
        # cv2.fillConvexPoly(imgBase, np.array(np.c_[x, y], dtype='int32'), 1)
        # imgMask = cv2.GaussianBlur(imgBase, (81, 81), 0)

        # imgBlur3D = np.ndarray(
        #     [self.height, self.width, 3], dtype='float')
        # imgBlur3D[:, :, 0] = imgMask
        # imgBlur3D[:, :, 1] = imgMask
        # imgBlur3D[:, :, 2] = imgMask
        # self.image_copy = (
        #     imgBlur3D*self.image + (1 - imgBlur3D)*self.image_copy).astype('uint8')

        img_base = np.zeros((self.height, self.width))
        cv2.fillConvexPoly(img_base, np.array(
            np.c_[x, y], dtype='int32'), 1)
        img_mask = cv2.GaussianBlur(
            img_base, (ksize_h, ksize_w), 0)  # 51,51 81,81
        img_blur_3d = np.ndarray(
            [self.height, self.width, 3], dtype='float')
        img_blur_3d[:, :, 0] = img_mask
        img_blur_3d[:, :, 1] = img_mask
        img_blur_3d[:, :, 2] = img_mask
        self.image_copy = (
            img_blur_3d * self.image + (1 - img_blur_3d) * self.image_copy).astype('uint8')