import numpy as np

class KeyFrame(object):
    def __init__(self, keyframeid = None, pointList = None, insertframeid = None, newpointstart = None ):
        if keyframeid is not None:
            self.keyframeid = keyframeid
        else:
            self.keyframeid = None

        '''pointList store the mapPoint idx that contains in the keyframe'''
        if pointList is not None:
            self.pointList = list(tuple(pointList))
        else:
            self.pointList = list([])

        if insertframeid is not None:
            self.insertframeid = insertframeid
        else:
            print("To add a keyframe, you need to add the insertframeid to perform BA")

        if newpointstart is not None:
            self.newpointstart = newpointstart
        else:
            self.newpointstart = 0

    def generateDescriptors(self, map):
        if self.pointList is None:
            print("[Error] There is no keypoint stored in keyframe")
            return False

        des = []
        for i in self.pointList:
            des.append(map.pointCloud[i].descriptors)

        return np.array(des)

    def updatePointList(self, pointList):
        '''pointList is a list that contains the idx in point cloud'''
        for p in pointList:
            self.pointList.append(p)

        print("[Keyframe] Update Keyframe NO. ", self.keyframeid, " Contains keypoints: ", len(self.pointList) )



