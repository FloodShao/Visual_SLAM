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

matches = VO.featureMatches(frame1.descriptors[:400], frame2.descriptors)
#VO.plotMatches(VO.frameStruct[0], VO.frameStruct[1], matches)

matchedPoints1, matchedPoints2 = VO.generateMatchedPoints(frame1, frame2, matches)

#print(matchedPoints1)
R, t = VO.findRelativePose(matchedPoints1, matchedPoints2, frame1.cameraParams)

frame1.R_w = np.eye(3)
frame1.t_w = np.zeros((3, 1))
frame2.R_w = R
frame2.t_w = t

#print(frame1.R_w, frame2.R_w)
points, pointIdx = VO.triangulation(frame1, frame2, matchedPoints1, matchedPoints2, matches)

start_count, end_count = VO.updatePointCloud(points, pointIdx, frame1, frame2)
print(start_count, ",  ", end_count)
print(len(VO.map.pointCloud))

#print(VO.map.pointCloud[0].descriptors)
#print(frame1.descriptors[pointIdx[0]])

#print(frame1.descriptors.shape)
#print(VO.generatePointCloudDescriptors().shape)

image3 = cv2.imread('../data/3.png')
frame3 = VO.addframe(image3)

des = VO.generatePointCloudDescriptors()
#print(des)


rvec, tvec, inliers = VO.newframePnP(frame3)

print(rvec,'\n', tvec)
R = util.rvect2Rmat(rvec)
print(R)

#print(inliers)
#print(len(VO.map.pointCloud))

theta = math.acos((R[0,0] + R[1,1] + R[2,2] -1) /2)
D, V = np.linalg.eig(R)
print(D)
print(V)
for i in range(len(D)):
    if D[i] == 1:
        print("my", theta * V[i])

v, jacobian = cv2.Rodrigues(R)
print(v)



