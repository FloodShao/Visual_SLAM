import numpy as np
from visualOdometry import VO
from map import Map
import cv2
import util

map = Map()
VO = VO(map)

image1 = cv2.imread('../data/capture_images_35.jpg')
image2 = cv2.imread('../data/capture_images_37.jpg')

frame1 = VO.addframe(image1)
frame2 = VO.addframe(image2)

VO.initilization(frame1, frame2)

print('frame1_R:', util.rvect2Rmat(frame1.r_w))
print('frame1_t:', frame1.t_w)
print('frame2_R:', util.rvect2Rmat(frame2.r_w))
print('frame2_t:', frame2.t_w)

image3 = cv2.imread('../data/capture_images_38.jpg')
frame3 = VO.addframe(image3)
VO.newframePnP(frame3)
print('frame3_R:', util.rvect2Rmat(frame3.r_w))
print('frame3_t:', frame3.t_w)


