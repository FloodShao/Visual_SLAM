from mapPoint import MapPoint
from keyframe import KeyFrame
from config_default import compare_keyframe

class Map(object):

    def __init__(self, pointCloud = None, keyframeset = None):
        if pointCloud is not None:
            self.pointCloud = pointCloud
        else:
            self.pointCloud = []

        if keyframeset is not None:
            self.keyframeset = keyframeset
        else:
            self.keyframeset = []

    def addMapPoint(self, mappoint):
        '''
        point = MapPoint( id = len(self.pointCloud), point3d=mappoint.point3d,
                          viewedframeid=mappoint.viewedframeid, descriptors=mappoint.descriptors)
        '''
        self.pointCloud.append(mappoint)

    def addKeyFrame(self, frame, end_count = 0):
        '''first we need to find out weather this frame is a keyframe'''
        temp_pointlist = frame.inliers
        flag = 0
        if not self.keyframeset: # for the first keyframe
            keyframe = KeyFrame(keyframeid=len(self.keyframeset), pointList=temp_pointlist, insertframeid=frame.id)
            self.keyframeset.append(keyframe)
            print("[Keyframe] Add a keyframe, keyframeid:", keyframe.keyframeid, "contains points:",
                  len(keyframe.pointList))
            flag = 1
        else:
            count = 0
            for kf in self.keyframeset[-1 : -(compare_keyframe['neighbor_keyframe'] +1) : -1]:
                '''We only compare with the nearest keyframe'''

                commonlist = list(set(temp_pointlist).intersection(set(kf.pointList)))

                if len(commonlist) > compare_keyframe['intersection_ratio'] * len(kf.pointList):
                    '''if there is one frame that has over 50% intersection with one frame, 
                    then we do not add the keyframe'''
                    count = count +1

            if count == 0:
                keyframe = KeyFrame(keyframeid=len(self.keyframeset), pointList=temp_pointlist, insertframeid=frame.id, newpointstart=end_count)
                self.keyframeset.append(keyframe)
                print("[Keyframe] Add a keyframe, keyframeid:", keyframe.keyframeid)
                flag = 1

        if(flag == 0):
            return False
        else:
            return True


    def updatePointCloud(self, points, pointIdx, frame1, frame2, matchedPoints1, matchedPoints2):
        """
        This function should be used after triangulation, to add more point in the map
        :param points: Nx3 triangulated points
        :param pointIdx: list of the feature index in frame1
        :param frame1: frame1 that used for triangulation
        :param frame2: frame2 that used for triangulation
        :return: updated pointCloud idx
        """

        start_count = len(self.pointCloud)

        i = 0
        for p in points:
            point = MapPoint(id = len(self.pointCloud), point3d=p, viewedframeid= None, descriptors=frame1.descriptors[pointIdx[i]])
            '''update the observation'''
            point.updateMapPoint(frame1.id, matchedPoints1[i])
            point.updateMapPoint(frame2.id, matchedPoints2[i])

            self.addMapPoint(point)
            i = i+1

        end_count = len(self.pointCloud)

        return start_count, end_count






