import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import util
from config_default import CameraIntrinsics


def project(points_3d, frameArgs):
    """
    This function do the projection from the 3D points to frame 2D pixel points.
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

        R = util.rvect2Rmat(frameArgs[idx][0:3])
        t = frameArgs[idx][3:6].reshape(3, 1)

        p_3d = points_3d[idx].reshape(3, 1) #convert to column
        p_CA = np.dot(R, p_3d) + t

        u_c = float(p_CA[0] / p_CA[2])
        v_c = float(p_CA[1] / p_CA[2])

        u_pixel = f * u_c + cx
        v_pixel = f * v_c + cy

        points_2d.append([u_pixel, v_pixel])

    return np.array(points_2d)


def reprojetion_error(params, points_2d_observe, n_frames, n_points, frame_indices, point_indices):
    """
    This function calculate the reproject error
    :param params: [frame_params, point_params]
    :param points_2d_observe: nx2, observed points_2d pixel
    :param n_frames: number of frames need to update
    :param n_points: number of points need to update
    :param frame_indices: nx1, indicate the frame indice for each observation
    :param point_indices: nx1. indicate the points3d indice for each observation
    :return: the reprojection error
    """

    frameArgs = params[: n_frames * 6].reshape( (n_frames, 6))
    points_3d = params[n_frames * 6 : ].reshape( (n_points, 3))

    points_2d_project = project(points_3d[point_indices], frameArgs[frame_indices])
    return (points_2d_project - points_2d_observe).ravel()


def bundle_adjustment_sparsity(n_frames, n_points, frame_indices, point_indices):
    """
    Construct a sparse matrix for optimize the least square problem
    :param n_frames: the number of included frames
    :param n_points: the number of included points_3d
    :param frame_indices: a matrix (n*1) includes frame_indices that correspond to points_2d
    :param point_indices: a matrix (n*1) includes point_indices that correspond to points_2d
    :return: a sparse matrix A
    """
    m = frame_indices.size * 2
    n = n_frames * 6 + n_points * 3
    A = lil_matrix((m,n), dtype = int)

    i = np.arange(frame_indices.size)

    '''We do not optimized the frame params that is fixed'''
    index = np.where(frame_indices == 0)
    j = np.delete(i, index)
    frame_i = np.delete(frame_indices, index)

    for s in range(6):
        A[2 * j, frame_i * 6 + s] = 1
        A[2 * j + 1, frame_i * 6 + s] = 1

    for s in range(3):
        A[2 * i, n_frames * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_frames * 6 + point_indices * 3 + s] = 1

    return A

def recoverResults(x, n_frames, n_points):
    frameArgs = x[ : 6 * n_frames]
    points = x[ 6*n_frames : ]

    frameArgs = frameArgs.reshape( (n_frames, 6) )
    points = points.reshape( (n_points, 3) )

    return frameArgs, points

