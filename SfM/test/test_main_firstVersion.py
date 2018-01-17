from visualOdometry import VO
import cv2
import os
import numpy as np
from map import Map
from config_default import file_dir

associate_file = file_dir['data_dir_tube']
try:
    fh = open(associate_file, 'r')
    assert os.path.getsize(associate_file)
except IOError:
    print("[Error] Failed to open the data_dir")
except AssertionError:
    print("[Warning] There is no data dir in", associate_file, "file!\n")


if __name__ == '__main__':

    '''construct the data structure and VO'''
    mymap = Map()
    myVO = VO(mymap)

    for line in fh.readlines():
        line = line.strip()
        image = cv2.imread(line)

        """
        VO.addframe function will proceed (1)feature detection
        (2)add frame to VO.frameStruct, frame id is the length of frameStruct 
        """
        curframe = myVO.addframe(image)

        if len(myVO.frameStruct) < 2:
            '''the default pose is eye(3) and zeros(3,1)'''
            continue #There is only 1 frame in the frameStruct
        else:
            if len(myVO.frameStruct) == 2: #proceed the initialization
                myVO.initilization(myVO.frameStruct[0], myVO.frameStruct[1])
                '''Now we add the fist keyframe'''
            else:
                '''a new frame added, need to update R_w, t_w, inliers'''
                try:
                    assert myVO.newframePnP(curframe) #update the R_w, t_w, and inliers
                except AssertionError:
                    print("[Warning] Failed to solve PnP for curframe, frame id: ", curframe.id)

                '''update the frameStructure'''
                myVO.updateframe(curframe)
                '''check whether need to add a keyframe'''
                if myVO.map.addKeyFrame(curframe):
                    '''add a new keyframe, need to perform triangulation to add more pointCloud'''
                    '''here curframe is queryIdx, and prevframe is trainIdx'''
                    prevframe = myVO.frameStruct[curframe.id-1]
                    '''(1) feature matching with prevframe'''
                    matches = myVO.featureMatches(curframe.descriptors, prevframe.descriptors)
                    matchedPoints1, matchedPoints2 = myVO.generateMatchedPoints(curframe, prevframe, matches)
                    '''(2)triangulation'''
                    points, pointIdx = myVO.triangulation(curframe, prevframe, matchedPoints1, matchedPoints2, matches)
                    '''(3)updatePointCloud'''
                    start_count, end_count = myVO.updatePointCloud(points, pointIdx, curframe, prevframe)
                    newly_keypoint_Idx = list(range(start_count, end_count))
                    '''(4)updatekeyframe'''
                    myVO.map.keyframeset[-1].updatePointList(newly_keypoint_Idx)


    fh.close()

    point_cloud = []
    for p in myVO.map.pointCloud:
        point_cloud.append(list(p.point3d))

    point_cloud_mat = np.array(point_cloud).transpose()

    np.save("./data/capsule_point_cloud_1_11.npy", point_cloud_mat)
    sio.savemat("./data/capsule_point_cloud_1_11.mat", point_cloud_mat)