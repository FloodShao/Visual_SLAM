from visualOdometry import VO
from map import Map
import cv2
import numpy as np
import util


"""check the reprojection
"""
map = Map()
VO = VO(map)


image1 = cv2.imread('../data/1.png')
image2 = cv2.imread('../data/2.png')

frame1 = VO.addframe(image1)
frame2 = VO.addframe(image2)

matches = VO.featureMatches(frame1.descriptors[:400], frame2.descriptors)
#VO.plotMatches(VO.frameStruct[0], VO.frameStruct[1], matches)

matchedPoints1, matchedPoints2 = VO.generateMatchedPoints(frame1, frame2, matches)

#print(matchedPoints1)
R, t = VO.findRelativePose(matchedPoints1, matchedPoints2, frame1.cameraParams)

frame1.updateframe( np.eye(3), np.zeros((3, 1)), None)
frame2.updateframe( R, t, None)

#print(frame1.R_w, frame2.R_w)
points, pointIdx = VO.triangulation(frame1, frame2, matchedPoints1, matchedPoints2, matches)

start_count, end_count = VO.updatePointCloud(points, pointIdx, frame1, frame2)
f_inliers = []
for p in VO.map.pointCloud[start_count: end_count]:
    f_inliers.append(p.id)

frame1.updateframe( None, None, f_inliers)
frame2.updateframe( None, None, f_inliers)

VO.map.addKeyFrame(frame1)

#print(VO.map.pointCloud[0].descriptors)
#print(frame1.descriptors[pointIdx[0]])

#print(frame1.descriptors.shape)
#print(VO.generatePointCloudDescriptors().shape)

image3 = cv2.imread('../data/3.png')
frame3 = VO.addframe(image3)

des = VO.generatePointCloudDescriptors()
#print(des)


rvec, tvec, inliers = VO.newframePnP(frame3)

f3_inliers = []

for i in inliers:
    f3_inliers.append(i[0])
frame3.updateframe(R_w, t_w, f3_inliers)

#print(f3_inliers)

flag = VO.map.addKeyFrame(frame3)
print(flag)


cv2.waitKey()


