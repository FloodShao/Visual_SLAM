import cv2
import numpy as np
import util
from frame import Frame
from map import Map
from config_default import triangulation_relate
import bundleAdjustment as BA
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

class VO(object):

    def __init__(self, map = None, frameStruct = None ):
        if Map is not None:
            self.map = map
        else:
            self.map = Map() #import Map, create the object


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
        search_params = dict(checks = 500)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(refDescriptors, curDescriptors, k = 2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
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
                                 method=cv2.RANSAC, prob=0.9, threshold=1e-2)
        '''
        U, sigma, VT = np.linalg.svd(E)
        sigma = [1, 1, 0]
        E = U * sigma * VT
        '''
        #print('mask', mask)

        #R1, R2, t = np.zeros((3, 3)), np.zeros((3,3)), np.zeros((3, 1))
        #cv2.decomposeEssentialMat(E, R1, R2, t)

        R, t = np.zeros((3, 3)), np.zeros((3, 1))
        cv2.recoverPose(E, matchedPoints1, matchedPoints2, cameraParameters, R, t, mask)

        '''

        pts1 = util.pixel2camera(matchedPoints1)
        pts2 = util.pixel2camera(matchedPoints2)

        f1_R = np.array([[1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]])
        f2_R = R

        f1_t = np.array([[0.], [0.], [0.]])
        f2_t = t

        points4d = np.zeros((4, len(matchedPoints1)))
        pMatrix1 = np.hstack( (f1_R, f1_t ) )
        pMatrix2 = np.hstack( (f2_R, f2_t ) )

        cv2.triangulatePoints(pMatrix1, pMatrix2, pts1.transpose()[:2, :], pts2.transpose()[:2, :], points4d)

        points4d = points4d.transpose()
        count = 0
        for i in range(points4d.shape[0]):
            if points4d[i, 2]/points4d[i, 3] < 0:
                count += 1

        print(count, '/', points4d.shape[0])
        
        if count < 0.5*points4d.shape[0]:
            return R, t
        else:
            return R, -t
        '''
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
        if(frame1.r_w is None or frame2.r_w is None or frame1.t_w is None or frame2.t_w is None):
            print("**Missing r or t. Please estimate the pose of frame1 or frame2 first!**")
            return False

        pts1 = util.pixel2camera(matchedPoints1, frame1.cameraParams)
        pts2 = util.pixel2camera(matchedPoints2, frame2.cameraParams)

        f1_R = util.rvect2Rmat(frame1.r_w)
        f2_R = util.rvect2Rmat(frame2.r_w)

        points4d = np.zeros( (4, len(matchedPoints1)) ) #triangulate in homogeneous coordinates
        pMatrix1 = np.hstack( (f1_R, frame1.t_w.reshape(3, 1)) )
        pMatrix2 = np.hstack( (f2_R, frame2.t_w.reshape(3, 1)) )

        #print(pMatrix1.shape)
        #print(matchedPoints1.shape)

        cv2.triangulatePoints(pMatrix1, pMatrix2, pts1.transpose()[:2, :], pts2.transpose()[:2, :], points4d)

        #mind that points4d is 4xN
        points4d = points4d.transpose()

        points3d = []
        pointIdx = []

        count = 0
        for i in range(points4d.shape[0]):
            point_3d = points4d[i, :3] / points4d[i, 3]  #mind the index of array
            '''we observe some constructed points have negative depth and those points that too far away, clear them out'''
            '''and np.linalg.norm(point_3d) < triangulation_relate['dis_threshold']'''
            if(point_3d[2] > 0 and np.linalg.norm(point_3d - frame1.t_w) < triangulation_relate['dis_threshold']):
                points3d.append(point_3d)
                pointIdx.append(matches[i].queryIdx)
                count += 1


        print('triangulation:', count, '/', points4d.shape[0])
        return np.array(points3d), pointIdx


    def generatePointCloudDescriptors(self):
        """
        This function generates the point descriptors from keyframeset
        :return:
        """
        if self.map.keyframeset is None:
            print("[Error] There is no Keyframe stored")

        #print(len(self.map.pointCloud))
        #print(self.map.pointCloud[0].descriptors.shape[0])

        descriptors = []
        point_list = []

        '''
        for kf in self.map.keyframeset:
            point_list.extend(kf.pointList)

        point_list = list(set(point_list))
        for p in point_list:
            descriptors.append(self.map.pointCloud[p].descriptors)
        '''

        for p in self.map.pointCloud[-1000:-1]:
            descriptors.append(p.descriptors)
            point_list.append(p.id)

        return np.array(descriptors), point_list

    def newframePnP(self, frame):
        """
        Proceed the solution for new frame PnP
        :param frame: newly added frame, with id, featurePoints, descriptors, cameraParams
        :return: if successful, return R_w, t_w, and inliers; if unsuccessful return False
        """
        des_Map, point_list = self.generatePointCloudDescriptors()

        matches = self.featureMatches(des_Map, frame.descriptors )

        objectPoints = []
        imagePoints = []
        matchpointid = []
        print('[newframePnP] matches: ', len(matches))
        for m in matches:
            objectPoints.append( self.map.pointCloud[point_list[m.queryIdx]].point3d )
            imagePoints.append( frame.featurePoints[m.trainIdx].pt )

        objectPoints = np.array(objectPoints)
        imagePoints = np.array(imagePoints)
        R_vector = self.frameStruct[frame.id-1].r_w
        t_vector = self.frameStruct[frame.id-1].t_w
        distCoeffs = None
        inliers = np.array([])
        flag, R_vector, t_vector, inliers = cv2.solvePnPRansac(objectPoints, imagePoints, frame.cameraParams, distCoeffs, R_vector, t_vector,
                           useExtrinsicGuess = False, iterationsCount = 100, reprojectionError = 8.0, confidence = 0.7,
                           inliers = inliers, flags=cv2.SOLVEPNP_DLS)

        flag, R_vector, t_vector, inliers = cv2.solvePnPRansac(objectPoints, imagePoints, frame.cameraParams, distCoeffs, R_vector, t_vector,
                           useExtrinsicGuess = True, iterationsCount = 100, reprojectionError = 2.0, confidence = 0.8,
                           inliers = inliers, flags=cv2.SOLVEPNP_ITERATIVE)

        f_inliers = []
        if inliers is None:
            return False

        for inl in inliers:
            f_inliers.extend(inl.tolist()) #mind that the inliers contains other parameters

        if(flag == True):
            frame.r_w = R_vector
            frame.t_w = t_vector
            frame.inliers = f_inliers
            print("Solved PnP for frame No. ", frame.id, " frame inliers: ", len(frame.inliers))
            return frame.r_w, frame.t_w, frame.inliers
        else:
            return False


    def initilization(self, frame1, frame2):

        '''(1)feature matches'''
        matches = self.featureMatches(frame1.descriptors, frame2.descriptors)
        #VO.plotMatches(frame1, frame2, matches)

        matchedPoints1, matchedPoints2 = self.generateMatchedPoints(frame1, frame2, matches)

        '''(2)find relative pose of frame2'''
        R, t = self.findRelativePose(matchedPoints1, matchedPoints2, frame1.cameraParams)
        '''mind that here R is 3x3 matrix, t is 3x1 vector'''
        f1_R = util.rvect2Rmat(frame1.r_w)
        f2_R = np.dot(R, f1_R)
        frame2.r_w = util.Rmat2rvec(f2_R)
        frame2.t_w = ( frame1.t_w.reshape(3,1) + np.dot(f1_R, t) )
        self.updateframe(frame2)

        '''(3)triangulation'''
        points, pointIdx = self.triangulation(frame1, frame2, matchedPoints1, matchedPoints2, matches)

        '''(4)updatePointCloud'''

        start_count, end_count = self.map.updatePointCloud(points, pointIdx, frame1, frame2, matchedPoints1, matchedPoints2)

        '''(5)addkeyframe'''
        f_inliers = []
        for p in self.map.pointCloud[start_count: end_count]:
            f_inliers.append(p.id)

        frame1.updateframe(None, None, f_inliers)
        frame2.updateframe(None, None, f_inliers)

        self.map.addKeyFrame(frame1)


    def localBA(self, start_frame, end_frame, start_point, end_point):
        """
        Do local BA, with consecutive frames and consecutive points
        :param start_frame: the start index of the frameStruct, however, the pose of the first frame is either fixed or
        already optimized. So we do not optimize this frame
        :param end_frame: the end index of the frameStruct
        :param start_point: ths start index of the point cloud
        :param end_point: tthe total length of the point cloud
        :return: optimized frame pose and point 3d
        """

        '''Prepare the optimized framework'''
        points_3d = []
        points_2d_observe = []
        point_indices = []
        frame_indices = []

        for p in self.map.pointCloud[start_point : end_point]:
            points_3d.append(p.point3d)
            for i in range(len(p.viewedframeid)):
                if p.viewedframeid[i] in range(start_frame, end_frame+1):
                    '''Double check the observation has corresponding frame'''
                    '''For each local BA, the first frame we optimize is frame0'''
                    frame_indices.append(p.viewedframeid[i] - start_frame)
                    points_2d_observe.append(p.projection_inframe[i])
                    point_indices.append(len(points_3d)-1)

        frameArgs = []
        for fi in range(start_frame, end_frame+1):
            r = self.frameStruct[fi].r_w
            t = self.frameStruct[fi].t_w
            frameArgs.append( [r[0], r[1], r[2], t[0], t[1], t[2]] )

        n_frames = len(range(start_frame, end_frame)) +1
        n_points = len(points_3d)

        A = BA.bundle_adjustment_sparsity(n_frames, n_points, np.array(frame_indices), np.array(point_indices))

        x0 = np.hstack( (np.array(frameArgs).ravel(), np.array(points_3d).ravel()) )

        res = least_squares(BA.reprojetion_error, x0, jac_sparsity=A,
                            verbose=2, x_scale='jac', ftol=1e-2, method='trf',
                            args=(points_2d_observe, n_frames, n_points, frame_indices, point_indices))

        frameArgs_opt, points_3d_opt = BA.recoverResults(res.x, n_frames, n_points)

        i = 0
        for fi in range(start_frame, end_frame+1):
            self.frameStruct[fi].r_w = frameArgs_opt[i, 0:3].reshape((3, 1))
            self.frameStruct[fi].t_w = frameArgs_opt[i, 3:6].reshape((3, 1))
            i += 1
            print("optimize frame id : ", fi)

        i = 0
        for pi in range(start_point, end_point):
            self.map.pointCloud[pi].points3d = points_3d_opt[i]


















