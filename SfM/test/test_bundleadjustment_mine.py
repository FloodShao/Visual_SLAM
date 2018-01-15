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





VO = VO()
image1 = cv2.imread('../data/capture_images_100.jpg')
image2 = cv2.imread('../data/capture_images_101.jpg')

frame1 = VO.addframe(image1)
frame2 = VO.addframe(image2)

VO.initilization(frame1, frame2)

points_3d = []
point_indices = []
camera_indices = []
points_2d_observe = []
for p in VO.map.pointCloud:
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
    r = util.Rmat2rvec( VO.frameStruct[fi].R_w ).reshape(1, 3)
    t = VO.frameStruct[fi].t_w.reshape(1, 3)
    cameraArgs.append(np.hstack((r, t)).tolist()[0] )

#points_2d = project(np.array(points_3d_corres), np.array(cameraArgs))
#points_2d_observe = np.array(points_2d_observe)

n_cameras = len(camera)
n_points = len(points_3d)


A = bundle_adjustment_sparsity(n_cameras, n_points, np.array(camera_indices), np.array(point_indices))

x0 = np.hstack( (np.array(cameraArgs).ravel(), np.array(points_3d).ravel() ) )
f0 = fun(x0, points_2d_observe, n_cameras, n_points, camera_indices, point_indices)

res = least_squares(
    fun, x0, jac_sparsity=A,
    verbose=2, x_scale='jac', ftol=1e-4, method='trf',
    args=(points_2d_observe, n_cameras, n_points, camera_indices, point_indices)
)

plt.figure(1)
plt.subplot(211)
plt.plot(f0)
plt.subplot(212)
plt.plot(res.fun)
plt.show()









