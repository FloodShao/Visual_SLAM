from visualOdometry import VO
from structurePointCloud import StructurePointCloud
import cv2
import os
import numpy as np
from map import Map
from config_default import file_dir
import matplotlib.pyplot as plt
import scipy.io as sio


def project(frame, points_3d):
    """
    Project a point to the frame pixel coordinate
    :param frame: frame object
    :param points_3d: points_3d list
    :return: points_2d list
    """
    R = frame.R_w
    t = frame.t_w
    f = frame.cameraParams[0, 0]
    cx = frame.cameraParams[0, 2]
    cy = frame.cameraParams[1, 2]

    points_idx = 0
    index = []

    points_2d = []
    for p in points_3d:
        p_mat = np.array(p).reshape(3, 1)
        P = np.dot(R, p_mat) + t
        u_c = float(P[0]/P[2])
        v_c = float(P[1]/P[2])

        u_s = f * u_c + cx
        v_s = f * v_c + cy

        if u_s > frame.shape[0] or u_s < 0 or v_s > frame.shape[1] or v_s < 0:
            points_idx += 1
            continue

        points_2d.append([u_s, v_s])
        index.append(points_idx)
        points_idx += 1


    return points_2d, index




VO = VO()

image1 = cv2.imread('./data/capture_images_100.jpg')
image2 = cv2.imread('./data/capture_images_101.jpg')

frame1 = VO.addframe(image1)
frame2 = VO.addframe(image2)

VO.initilization(frame1, frame2)



point_cloud = []
points0 = []
points1 = []
for p in VO.map.pointCloud:
    point_cloud.append(list(p.point3d))
    for i in range(len(p.viewedframeid)):
        if p.viewedframeid[i] == 0:
            points0.append(p.projection_inframe[i])
        else:
            points1.append(p.projection_inframe[i])

points_2d, index= project(frame1, point_cloud)

print(len(point_cloud[0]))
print(len(points_2d))

for i in range(len(points_2d)):
    print(points_2d[i], "    ", points0[index[i]])

cv2.waitKey()


