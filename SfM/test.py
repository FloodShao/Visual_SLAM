from frame import Frame
from visualOdometry import VO
import cv2
import numpy as np
import util
from config_default import CameraIntrinsics


VO = VO()

image1 = cv2.imread('./data/1.png')
image2 = cv2.imread('./data/2.png')
"""
frame1 = Frame(0)
frame2 = Frame(1)

des1 = frame1.featureDetection_SIFT(image1)
des2 = frame2.featureDetection_SIFT(image2)

matches = frame2.featureMatches(des1, des2)
frame2.plotMatches(frame1, frame2, matches)
"""
frame1, des1 = VO.addframe(image1)
frame2, des2 = VO.addframe(image2)

matches = VO.featureMatches(des1, des2)
#VO.plotMatches(VO.frameStruct[0], VO.frameStruct[1], matches)

matchedPoints1, matchedPoints2 = VO.generateMatchedPoints(frame1, frame2, matches)

#print(matchedPoints1)
R, t = VO.findRelativePose(matchedPoints1, matchedPoints2, frame1.cameraParams)

frame1.R_w = np.eye(3)
frame1.t_w = np.zeros((3,1))
frame2.R_w = R
frame2.t_w = t

#print(frame1.R_w, frame2.R_w)
points = VO.triangulation(frame1, frame2, matchedPoints1, matchedPoints2)


A = np.zeros(points.shape)
for i in range(points.shape[0]):
    A[i, :2] = points[i, :2] / points[i, 2]
    A[i, 2] = points[i, 2]
print(A)
B = util.pixel2camera(matchedPoints1, frame1.cameraParams)

print(B)

cv2.waitKey()