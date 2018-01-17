import numpy as np
import math
import cv2

def world2camera(point_world, frame):
    """

    :param point_world: Nx3 array, point in world coordinates
    :param frame: frame properties
    :return: point_camera
    """
    point_camera = np.dot(frame.R_w, point_world.transpose()) + frame.t_w

    return point_camera.transpose()


def camera2pixel(point_camera, cameraParams):
    """

    :param point_camera: Nx3 array, point in camera coordinates
    :param cameraParams: 3x3 array, camera intrinsics
    :return: point_pixel Nx2
    """
    point_frame = np.dot(cameraParams, point_camera.transpose()).transpose() #Nx3
    point = np.zeros( (point_frame.shape[0], 2) ) #Nx2
    for i in range(point_frame.shape[0]):
        point[i, :] = point_frame[i, :2] / point_frame[i, 2]

    return point

def world2pixel(point_world, frame):
    point_camera = world2camera(point_world, frame)
    point = camera2pixel(point_camera, frame.cameraParams)

    return point

def pixel2camera(point, cameraParameters, depth = 1):
    point_camera = np.zeros( (point.shape[0], 3))
    for i in range(point_camera.shape[0]):

        point_camera[i, 0] = (point[i, 0] - cameraParameters[0, 2]) * depth / cameraParameters[0, 0]
        point_camera[i, 1] = (point[i, 1] - cameraParameters[1, 2]) * depth / cameraParameters[1, 1]
        point_camera[i, 2] = depth

    return point_camera

def skewMatrix( vector ):
    """
    Change a 3x1 vector to the corresponding skewMatrix
    :param vector: 3x1 vector
    :return: skewMatrix
    """
    skew = [[0, -vector[2], vector[1]],
            [vector[2], 0, -vector[0]],
            [-vector[1], vector[0], 0]]

    return np.array(skew)

def rvect2Rmat( rvec ):
    """
    Change the rotation vector to rotation matrix, recommend to use cv2.Rodrigues
    :param rvec: 3x1 input rotation vector
    :return: output rotation matrix
    """

    Rmat, J = cv2.Rodrigues(rvec)
    '''
    if rvec is None:
        print("[Error] No rvec input")
        return False

    theta = np.linalg.norm(rvec)
    rvec = rvec/theta

    Rmat = math.cos(theta) * np.eye(3) + \
           (1 - math.cos(theta)) * rvec * rvec.transpose() + \
           math.sin(theta) * skewMatrix(rvec)
    '''
    return Rmat

def Rmat2rvec( Rmat):
    """
    Provide 2 ways to convert, recommend cv2 method
    :param Rmat: 3x3 Rmat
    :return: rvec 1x3
    """
    '''method cv2'''
    rvec, Jacobian = cv2.Rodrigues(Rmat)

    '''method eigen vector decomposition'''
    '''
    theta = math.acos((R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2)
    D, V = np.linalg.eig(R)
    for i in range(len(D)):
        if D[i] == 1:
            rvec = np.array(theta * V[i]).transpose()
    '''

    return rvec




