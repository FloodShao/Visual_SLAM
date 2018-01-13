import cv2
import numpy as np
import util
from frame import Frame
from mapPoint import MapPoint
from map import Map
from structurePointCloud import StructurePointCloud

class VO(object):

    def __init__(self, map = None, structurePointCloud = None, frameStruct = None, ):
        if Map is not None:
            self.map = map
        else:
            self.map = Map() #import Map, create the object

        if structurePointCloud is not None:
            self.structurePointCloud = structurePointCloud
        else:
            self.structurePointCloud = StructurePointCloud()

        if frameStruct is not None:
            self.frameStruct = frameStruct  #list
        else:
            self.frameStruct = list([])


    def addframe(self, image):
        id = len(self.frameStruct) #start from 0
        frame = Frame(id)
        frame.featureDetection_SIFT(image) #update features and descriptors
        #frame.featureDetection_ORB(image)
        self.frameStruct.append(frame)
        print("Successfully add frame No.", id)
        return frame


    def updateframe(self, frame):
        self.frameStruct[frame.id] = frame
        print("Update frame NO. ", frame.id)


    def featureMatches(self, refDescriptors,  curDescriptors):

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(refDescriptors, curDescriptors, k = 2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        '''bf matching, need to use featureDetection_ORB, change in VO.addframe'''
        '''
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
        matches = bf.match(refDescriptors, curDescriptors)
        good_matches = []

        max_dis = 0.0
        min_dis = 10000.0
        for m in matches:
            if(m.distance < min_dis):
                min_dis = m.distance
            if(m.distance > max_dis):
                max_dis = m.distance

        for m in matches:
            if(m.distance <= max( 2*min_dis, 30.0) ):
                good_matches.append(m)
        '''
        return good_matches


    def plotMatches(self, frame1, frame2, matches):
        img1 = frame1.image
        img2 = frame2.image

        kp1 = frame1.featurePoints
        kp2 = frame2.featurePoints

        draw_params = dict(matchColor = (0,255,0), #draw matches in green color
                           singlePointColor = (255,0,0),
                           matchesMask = None, #draw only inliers
                           flags = 2)

        outimage = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
        cv2.imshow("matches", outimage)


    def generateMatchedPoints(self, frame1, frame2, matches):
        '''provide matchedPoints for function findRelativePose'''
        matchedPoints1, matchedPoints2 = np.zeros( (len(matches), 2) ), np.zeros( (len(matches), 2) )
        i = 0
        for m in matches:
            matchedPoints1[i, :] = frame1.featurePoints[m.queryIdx].pt

            matchedPoints2[i, :] = frame2.featurePoints[m.trainIdx].pt

            i = i+1
        return matchedPoints1, matchedPoints2


    def findRelativePose(self, matchedPoints1, matchedPoints2, cameraParameters):

        E, mask = cv2.findEssentialMat(matchedPoints1, matchedPoints2, cameraParameters,
                                 method=cv2.RANSAC, prob=0.999, threshold=1.0, mask = None)
        '''
        U, sigma, VT = np.linalg.svd(E)
        sigma = [1, 1, 0]
        E = U * sigma * VT
        '''

        R, t = np.zeros((3, 3)), np.zeros((3, 1))

        cv2.recoverPose(E, matchedPoints1, matchedPoints2, cameraParameters, R, t, mask = None)

        return R, t


    def triangulation(self, frame1, frame2, matchedPoints1, matchedPoints2, matches):
        """
        This function perform points triangulation using OpenCV package
        :param frame1: the reference frame, with R and t in world coordinates
        :param frame2: the current frame, with R and t in world coordinates
        :param matchedPoints1: Nx2 array of feature points in frame1
        :param matchedPoints2: Nx2 array of feature points in frame2
        :param matches: indicate the matching connection
        :return: if triangulation is successful, return points (3xN) in world coordinates & featureIdx in frame1; if not, return false
        """
        if(frame1.R_w is None or frame2.R_w is None or frame1.t_w is None or frame2.t_w is None):
            print("**Missing R or t. Please estimate the pose of frame1 or frame2 first!**")
            return False

        pts1 = util.pixel2camera(matchedPoints1, frame1.cameraParams)
        pts2 = util.pixel2camera(matchedPoints2, frame2.cameraParams)

        points4d = np.zeros( (4, len(matchedPoints1)) ) #triangulate in homogeneous coordinates
        pMatrix1 = np.hstack( (frame1.R_w, frame1.t_w) )
        pMatrix2 = np.hstack( (frame2.R_w, frame2.t_w) )


        #print(pMatrix1.shape)
        #print(matchedPoints1.shape)

        cv2.triangulatePoints(pMatrix1, pMatrix2, pts1.transpose()[:2, :], pts2.transpose()[:2, :], points4d)

        #mind that points4d is 4xN
        points4d = points4d.transpose()

        points3d = list([])
        pointIdx = list([])
        for i in range(points4d.shape[0]):
            point_3d = points4d[i, :3] / points4d[i, 3]  #mind the index of array
            '''we observe some constructed points have negative depth, we should clear them out'''
            if(point_3d[2] > 0):
                points3d.append(point_3d)
                pointIdx.append(matches[i].queryIdx)

        '''every time we do triangulation, we store the point'''
        for p in points3d:
            self.structurePointCloud.addpoint(p, 'b')

        print("**add to structure point, ", "Points: ", len(self.structurePointCloud.mappoint))
        return np.array(points3d), pointIdx

    def updatePointCloud(self, points, pointIdx, frame1, frame2, matchedPoints1, matchedPoints2):
        """
        This function should be used after triangulation, to add more point in the map
        :param points: Nx3 triangulated points
        :param pointIdx: list of the feature index in frame1
        :param frame1: frame1 that used for triangulation
        :param frame2: frame2 that used for triangulation
        :return: updated pointCloud idx
        """
        if self.map is None:
            temp_map = Map()
        else:
            temp_map = self.map

        start_count = len(temp_map.pointCloud)

        i = 0
        for p in points:
            point = MapPoint(id = None, point3d=p, viewedframeid= None, descriptors=frame1.descriptors[pointIdx[i]])
            point.updateMapPoint(frame1.id, matchedPoints1[i])
            point.updateMapPoint(frame2.id, matchedPoints2[i])

            temp_map.addMapPoint(point)
            i = i+1

        end_count = len(temp_map.pointCloud)

        self.map = temp_map
        return start_count, end_count

    def generatePointCloudDescriptors(self):
        if self.map.pointCloud is None:
            print("[Error] There is no pointCloud stored")

        #print(len(self.map.pointCloud))
        #print(self.map.pointCloud[0].descriptors.shape[0])

        descriptors = []
        for p in self.map.pointCloud:
            descriptors.append(p.descriptors)

        return np.array(descriptors)

    def newframePnP(self, frame):
        """
        Proceed the solution for new frame PnP
        :param frame: newly added frame, with id, featurePoints, descriptors, cameraParams
        :return: if successful, return R_w, t_w, and inliers; if unsuccessful return False
        """
        des_Map = self.generatePointCloudDescriptors()

        matches = self.featureMatches(des_Map, frame.descriptors )

        objectPoints = []
        imagePoints = []
        for m in matches:
            objectPoints.append( self.map.pointCloud[m.queryIdx].point3d )
            imagePoints.append( frame.featurePoints[m.trainIdx].pt )

        objectPoints = np.array(objectPoints)
        imagePoints = np.array(imagePoints)
        R_vector = np.array([])
        t_vector = np.array([])
        distCoeffs = None
        inliers = np.array([])
        flag, R_vector, t_vector, inliers = cv2.solvePnPRansac(objectPoints, imagePoints, frame.cameraParams, distCoeffs, R_vector, t_vector,
                           useExtrinsicGuess = False, iterationsCount = 100, reprojectionError = 8.0, confidence = 0.99,
                           inliers = inliers)
        f_inliers = []
        for inl in inliers:
            f_inliers.append(inl[0].T) #mind that the inliers contains other parameters

        if(flag == True):
            frame.R_w = util.rvect2Rmat(R_vector)
            frame.t_w = t_vector
            frame.inliers = f_inliers
            print("Solved PnP for frame No. ", frame.id, " frame inliers: ", len(frame.inliers))
            return frame.R_w, frame.t_w, frame.inliers
        else:
            return False


    def initilization(self, frame1, frame2):

        '''(1)feature matches'''
        matches = self.featureMatches(frame1.descriptors, frame2.descriptors)
        #VO.plotMatches(frame1, frame2, matches)

        matchedPoints1, matchedPoints2 = self.generateMatchedPoints(frame1, frame2, matches)

        '''(2)find relative pose of frame2'''
        R, t = self.findRelativePose(matchedPoints1, matchedPoints2, frame1.cameraParams)
        frame2.R_w = R * frame1.R_w
        frame2.t_w = frame1.t_w + np.dot(frame1.R_w , t)
        self.updateframe(frame2)

        '''(3)triangulation'''
        points, pointIdx = self.triangulation(frame1, frame2, matchedPoints1, matchedPoints2, matches)

        '''(4)updatePointCloud'''

        start_count, end_count = self.updatePointCloud(points, pointIdx, frame1, frame2, matchedPoints1, matchedPoints2)

        '''(5)addkeyframe'''
        f_inliers = []
        for p in self.map.pointCloud[start_count: end_count]:
            f_inliers.append(p.id)

        frame1.updateframe(None, None, f_inliers)
        frame2.updateframe(None, None, f_inliers)

        self.map.addKeyFrame(frame1)

















