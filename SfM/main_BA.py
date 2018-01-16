from visualOdometry import VO
import cv2
import os
import numpy as np
from config_default import file_dir
from config_default import CameraIntrinsics
import util

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

associate_file = file_dir['data_dir_tube']
try:
    fh = open(associate_file, 'r')
    assert os.path.getsize(associate_file)
except IOError:
    print("[Error] Failed to open the data_dir")
except AssertionError:
    print("[Warning] There is no data dir in", associate_file, "file!\n")


def project(points_3d, cameraArgs):
    """
    This function do the projection from the 3D points to frame 2D pixel points.
    1 3d-point project to multiple cameras
    As the point_3d provided is constructed from the frame, so the points must be seen from the frame.
    There is no need to test whether the point is within the frame
    :param cameraArgs: matrix(n*6), each cameraArg includes rvec(1*3) and tvec(1*3)
    :param point_3d: matrix(n*3)
    :return: the projection 2d pixel coordinates wrt the camera
    """
    '''we use the same camera intrinsics'''
    f = CameraIntrinsics['focal_length']
    cx = CameraIntrinsics['cx']
    cy = CameraIntrinsics['cy']

    points_2d = []

    for idx in range(len(points_3d)):

        R = util.rvect2Rmat(cameraArgs[idx][0:3])
        t = cameraArgs[idx][3:6].reshape(3, 1)

        p_3d = points_3d[idx].reshape(3, 1) #convert to column
        p_CA = np.dot(R, p_3d) + t

        u_c = float(p_CA[0] / p_CA[2])
        v_c = float(p_CA[1] / p_CA[2])

        u_pixel = f * u_c + cx
        v_pixel = f * v_c + cy

        points_2d.append([u_pixel, v_pixel])

    return np.array(points_2d)


def fun(params, points_2d_observe, n_cameras, n_points, camera_indices, point_indices):
    """
    This function calculate the reproject error
    :param params: [camera_params, point_params]
    :param points_2d_observe: observed points_2d pixel
    :param n_cameras:
    :param n_points:
    :param camera_indices:
    :param point_indices:
    :return:
    """

    cameraArgs = params[: n_cameras * 6].reshape( (n_cameras, 6))
    points_3d = params[n_cameras * 6 : ].reshape( (n_points, 3))

    points_2d_project = project(points_3d[point_indices], cameraArgs[camera_indices])
    return (points_2d_project - points_2d_observe).ravel()


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    """
    Construct a sparse matrix for optimize the least square problem
    :param n_cameras: the number of included frames
    :param n_points: the number of included points_3d
    :param camera_indices: a matrix (n*1) includes camera_indices that correspond to points_2d
    :param point_indices: a matrix (n*1) includes point_indices that correspond to points_2d
    :return: a sparse matrix A
    """
    m = camera_indices.size * 2
    n = n_cameras * 6 + n_points * 3
    A = lil_matrix((m,n), dtype = int)

    i = np.arange(camera_indices.size)
    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1 , camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

    return A


def recoverResults(x, n_cameras, n_points):
    cameraArgs = x[ : 6 * n_cameras]
    points = x[ 6*n_cameras : ]

    cameraArgs = cameraArgs.reshape( (n_cameras, 6) )
    points = points.reshape( (n_points, 3) )

    return cameraArgs[:, 3:6], points





if __name__ == '__main__':

    '''construct the data structure and VO'''
    # mymap = Map()
    # myStructPoint = StructurePointCloud()
    myVO = VO()

    for line in fh.readlines():
        line = line.strip()
        image = cv2.imread(line)

        """
        VO.addframe function will proceed (1)feature detection
        (2)add frame to VO.frameStruct, frame id is the length of frameStruct 
        """
        curframe = myVO.addframe(image)

        if len(myVO.frameStruct) < 2:
            '''the default pose is eye(3) and zeros(3,1)'''
            continue  # There is only 1 frame in the frameStruct
        else:
            if len(myVO.frameStruct) == 2:  # proceed the initialization
                myVO.initilization(myVO.frameStruct[0], myVO.frameStruct[1])
                '''Now we add the fist keyframe'''

            else:
                '''a new frame added, need to update R_w, t_w, inliers'''
                try:
                    assert myVO.newframePnP(curframe)  # update the R_w, t_w, and inliers
                except AssertionError:
                    print("[Warning] Failed to solve PnP for curframe, frame id: ", curframe.id)

                '''update the frameStructure'''
                myVO.updateframe(curframe)
                '''check whether need to add a keyframe'''
                if myVO.map.addKeyFrame(curframe):
                    '''add a new keyframe, need to perform triangulation to add more pointCloud'''
                    '''here curframe is queryIdx, and prevframe is trainIdx'''
                    prevframe = myVO.frameStruct[curframe.id - 1]
                    '''(1) feature matching with prevframe'''
                    matches = myVO.featureMatches(curframe.descriptors, prevframe.descriptors)
                    matchedPoints1, matchedPoints2 = myVO.generateMatchedPoints(curframe, prevframe, matches)
                    '''(2)triangulation'''
                    points, pointIdx = myVO.triangulation(curframe, prevframe, matchedPoints1, matchedPoints2, matches)
                    '''(3)updatePointCloud'''
                    start_count, end_count = myVO.updatePointCloud(points, pointIdx, curframe, prevframe,
                                                                   matchedPoints1, matchedPoints2)
                    newly_keypoint_Idx = list(range(start_count, end_count))
                    '''(4)updatekeyframe'''
                    myVO.map.keyframeset[-1].updatePointList(newly_keypoint_Idx)
                else:
                    prevframe = myVO.frameStruct[curframe.id - 1]
                    matches = myVO.featureMatches(curframe.descriptors, prevframe.descriptors)
                    matchedPoints1, matchedPoints2 = myVO.generateMatchedPoints(curframe, prevframe, matches)
                    points, pointIdx = myVO.triangulation(curframe, prevframe, matchedPoints1, matchedPoints2, matches)

    fh.close()

    points_3d =[]
    point_indices = []
    camera_indices = []
    points_2d_observe = []
    for p in myVO.map.pointCloud:
        points_3d.append(p.point3d)
        for i in range(len(p.viewedframeid)):
            camera_indices.append(p.viewedframeid[i])
            points_2d_observe.append(p.projection_inframe[i])
            point_indices.append(len(points_3d)-1)

    points_3d_corres = []
    for i in point_indices:
        points_3d_corres.append(points_3d[i])

    camera = list(set(camera_indices))
    cameraArgs = []
    for fi in camera:
        r = util.Rmat2rvec(myVO.frameStruct[fi].R_w).reshape(1, 3)
        t = myVO.frameStruct[fi].t_w.reshape(1, 3)
        cameraArgs.append(np.hstack((r, t)).tolist()[0])

    n_cameras = len(camera)
    n_points = len(points_3d)

    camera_indices_temp = []
    for c in camera_indices:
        camera_indices_temp.append(camera.index(c))

    camera_indices = camera_indices_temp

    A = bundle_adjustment_sparsity(n_cameras, n_points, np.array(camera_indices), np.array(point_indices))

    x0 = np.hstack((np.array(cameraArgs).ravel(), np.array(points_3d).ravel()))
    f0 = fun(x0, points_2d_observe, n_cameras, n_points, camera_indices, point_indices)

    res = least_squares(
        fun, x0, jac_sparsity=A,
        verbose=2, x_scale='jac', ftol=1e-3, method='trf',
        args=(points_2d_observe, n_cameras, n_points, camera_indices, point_indices)
    )

    plt.figure(1)
    plt.subplot(211)
    plt.plot(f0)
    plt.subplot(212)
    plt.plot(res.fun)
    plt.show()

    path, points = recoverResults(res.x, n_cameras, n_points)
    print(path.shape)
    print(points.shape)

    fig = plt.figure(1)


    x, y, z = points.transpose()[0], points.transpose()[1], points.transpose()[2]
    xp, yp, zp = path.transpose()[0], path.transpose()[1], path.transpose()[2]

    ax = Axes3D(fig)
    ax.scatter(x, y, z, 'b')
    ax.plot(xp, yp, zp, 'r')
    plt.show()

    fig1 = plt.figure(1)
    ax1 = Axes3D(fig1)
    ax1.plot(xp, yp, zp, 'r')

    plt.show()










