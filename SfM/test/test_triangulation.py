from visualOdometry import VO
import cv2
import numpy as np
import util
import math

"""check the reprojection
"""

VO = VO()

image1 = cv2.imread('../data/1.png')
image2 = cv2.imread('../data/2.png')

frame1 = VO.addframe(image1)
frame2 = VO.addframe(image2)

matches = VO.featureMatches(frame1.descriptors, frame2.descriptors)
VO.plotMatches(VO.frameStruct[0], VO.frameStruct[1], matches)

matchedPoints1, matchedPoints2 = VO.generateMatchedPoints(frame1, frame2, matches)

#print(matchedPoints1)
R, t = VO.findRelativePose(matchedPoints1, matchedPoints2, frame1.cameraParams)

frame1.R_w = np.eye(3)
frame1.t_w = np.zeros((3, 1))
frame2.R_w = R
frame2.t_w = t

#print(frame1.R_w, frame2.R_w)
points, pointIdx = VO.triangulation(frame1, frame2, matchedPoints1, matchedPoints2, matches)


A = np.zeros(points.shape)
for i in range(points.shape[0]):
    A[i, :2] = points[i, :2] / points[i, 2]
    A[i, 2] = points[i, 2]
print(A)
B = util.pixel2camera(matchedPoints1, frame1.cameraParams)
print(B)

count = 0
for i in points:
    if i[2] < 0:
        count = count + 1


print(count)
print(points.shape)
print(pointIdx)
print(len(pointIdx))



