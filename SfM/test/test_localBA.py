from visualOdometry import VO
import cv2
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import numpy as np
import util
from config_default import CameraIntrinsics
import matplotlib.pyplot as plt


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


def fun(params, prev_camera_param, points_2d_observe, n_cameras, n_points, camera_indices, point_indices):
    """
    This function provide the reprojection error for local BA
    :param params: the params that need to be optimize, (camera_params.ravel(), points_3d.ravel())
    :param prev_camera_param: prev_camera_prama (1*6), this camera should not be optimized, but should contribute to
    the reprojection error
    :param points_2d_observe: the array of observed 2d pixel point
    :param n_cameras: number of camera included in params(camera_params)
    :param n_points: number of points3d included in params(points_3d)
    :param camera_indices: has the same length as points_2d_observe, so it contains the prev_camera_indices
    :param point_indices: has the same length as points_2d_observe
    :return: the reprojection error
    """

    cameraArgs = params[: n_cameras * 6].reshape( (n_cameras, 6))
    points_3d = params[n_cameras * 6 : ].reshape( (n_points, 3))

    cameraArgs = np.vstack((prev_camera_param, cameraArgs))

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