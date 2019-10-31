# coding:utf-8
from __future__ import print_function
from __future__ import division

import os
import cv2 as cv
import numpy as np
import scipy.io as sio

data = sio.loadmat(
    os.path.join('data', '4s', 'Train',
                 'A01T_potential_30_35_1.mat'))['images_average']
label = sio.loadmat(os.path.join('data', '4s', 'Train',
                                 'A01T_label.mat'))['classlabel'] - 1

title_window = 'Interpolated Images'


def on_trackbar1(val):
    index = val // 1
    trial = cv.getTrackbarPos(trackbar_name2, title_window)
    cv.imshow(
        title_window,
        (data[trial, index, :, :, 0] - np.min(data[trial, index, :, :, 0])) /
        np.max((data[trial, index, :, :, 0] -
                np.min(data[trial, index, :, :, 0]))))


def on_trackbar2(val):
    trial = val // 1
    index = cv.getTrackbarPos(trackbar_name1, title_window)
    cv.imshow(
        title_window,
        (data[trial, index, :, :, 0] - np.min(data[trial, index, :, :, 0])) /
        np.max((data[trial, index, :, :, 0] -
                np.min(data[trial, index, :, :, 0]))))


cv.namedWindow(title_window, flags=cv.WINDOW_GUI_EXPANDED)
trackbar_name1 = 'Image index'
cv.createTrackbar(trackbar_name1, title_window, 0, data.shape[1] - 1,
                  on_trackbar1)
trackbar_name2 = 'Trial index'
cv.createTrackbar(trackbar_name2, title_window, 0, data.shape[0] - 1,
                  on_trackbar2)
# Show some stuff
on_trackbar1(0)
on_trackbar2(0)
# Wait until user press some key
cv.waitKey()
cv.destroyAllWindows()