from mapPoint import MapPoint
from keyframe import KeyFrame
from config_default import compare_keyframe

class Map(object):

    def __init__(self, pointCloud = None, keyframeset = None):
        if pointCloud is not None:
            self.pointCloud = pointCloud
        else:
            self.pointCloud = list([])

        if keyframeset is not None:
            self.keyframeset = keyframeset
        else:
            self.keyframeset = list([])

    def addMapPoint(self, mappoint):
        point = MapPoint( id = len(self.pointCloud), point3d=mappoint.point3d,
                          viewedframeid=mappoint.viewedframeid, descriptors=mappoint.descriptors)
        self.pointCloud.append(point)

    def addKeyFrame(self, frame):
        '''first we need to find out weather this frame is a keyframe'''
        temp_pointlist = frame.inliers
        flag = 0
        if not self.keyframeset: # for the first keyframe
            keyframe = KeyFrame(keyframeid=len(self.keyframeset), pointList=temp_pointlist)
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
                keyframe = KeyFrame(keyframeid=len(self.keyframeset), pointList=temp_pointlist)
                self.keyframeset.append(keyframe)
                print("[Keyframe] Add a keyframe, keyframeid:", keyframe.keyframeid)
                flag = 1

        if(flag == 0):
            return False
        else:
            return True







