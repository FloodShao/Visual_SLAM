import numpy as np
import cv2
from config_default import CameraIntrinsics

class Frame(object):

    def __init__(self, id = None, featurePoints = None, descriptors = None, cameraParams = None, R_w = None, t_w = None, image = None):
        """R_w: rotation matrix wrt the world coordinates.
           t_w: translation matrix wrt the world coordinates.
        """
        if id is not None:
            self.id = id
        else:
            self.id = None

        if featurePoints is not None:
            self.featurePoints = [featurePoints]  #feature lists
        else:
            self.featurePoints = None

        if descriptors is not None:
            self.descriptors = descriptors  #np.array
        else:
            self.descriptors = None

        if cameraParams is not None:  #waiting update for import configuration
            self.cameraParams = cameraParams  #np matrix 3*3, the camera intrinsics
        else:
            self.cameraParams = self.getCameraParams(CameraIntrinsics)

        if R_w is not None:
            self.R_w = R_w #rotation matrix 3*3
        else:
            self.R_w = np.array([ [1., 0., 0.],
                                  [0., 1., 0.],
                                  [0., 0., 1.]])

        if t_w is not None:
            self.t_w = t_w #translation matrix 3*1
        else:
            self.t_w = np.array([ [0.],
                                  [0.],
                                  [0.]])

        self.image = image
        self.inliers = []



    def featureDetection_SIFT(self, image):
        '''img is np.ndarray type'''
        '''test if the image is rgb'''
        if image is None:
            print("The image is None. \n")
            return False

        self.shape = image.shape
        self.image = image
        if len(image.shape) == 2:
            img = image
        else:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        '''use SURF algorithm to detect features, mind that in opencv3.0 you need to install opencv-contrib-python'''
        surf = cv2.xfeatures2d.SURF_create()
        featurePoints, descriptors = surf.detectAndCompute(img, None)  #return a list of featurePoints and descriptors matrix

        self.featurePoints = featurePoints
        self.descriptors = descriptors

    def featureDetection_ORB(self, image):
        if image is None:
            print("The image is None. \n")
            return False

        self.shape = image.shape
        self.image = image
        if len(image.shape) == 2:
            img = image
        else:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create()
        featurePoints, descriptors = orb.detectAndCompute(img, None)

        self.featurePoints = featurePoints
        self.descriptors = descriptors

    def plotFeatures(self):
        outimage = np.zeros(self.shape, np.uint8)
        cv2.drawKeypoints(self.image, self.featurePoints, outimage)
        cv2.imshow("Features", outimage)

    def plotFeaturestracked(self, featureIdx):
        outimage = np.zeros(self.shape, np.uint8)
        featureTracked = []
        for idx in featureIdx:
            featureTracked.append(self.featurePoints[idx])
        cv2.drawKeypoints(self.image, featureTracked, outimage)
        cv2.imshow("Features", outimage)


    def getCameraParams(self, CameraIntrinsics):
        K = np.zeros((3, 3))
        K[0, 0] = CameraIntrinsics['focal_length']
        K[0, 2] = CameraIntrinsics['cx']
        K[1, 1] = CameraIntrinsics['focal_length']
        K[1, 2] = CameraIntrinsics['cy']
        K[2, 2] = 1

        return K

    def updateframe(self, R_w, t_w, inliers):
        '''if the arguments is None, then do not update those items'''
        if R_w is not None:
            if(R_w.shape != (3,3)):
                print("[Error] frame.updateframe: R_w is not a 3x3 array" )
                return False
            else:
                self.R_w = R_w

        if t_w is not None:
            if(t_w.shape != (3,1)):
                print("[Error] frame.updateframe: t_w is not a 3x1 array" )
                return False
            else:
                self.t_w = t_w

        if inliers is not None:
            for inl in inliers:
                self.inliers.append( inl ) #the list that include the PnP inliers list

        return True








