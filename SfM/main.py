from visualOdometry import VO
from structurePointCloud import StructurePointCloud
import cv2
import os
import numpy as np
from map import Map
from config_default import file_dir
import matplotlib.pyplot as plt
import scipy.io as sio

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
    #mymap = Map()
    #myStructPoint = StructurePointCloud()
    myVO = VO()


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
                    start_count, end_count = myVO.updatePointCloud(points, pointIdx, curframe, prevframe, matchedPoints1, matchedPoints2)
                    newly_keypoint_Idx = list(range(start_count, end_count))
                    '''(4)updatekeyframe'''
                    myVO.map.keyframeset[-1].updatePointList(newly_keypoint_Idx)
                else:
                    prevframe = myVO.frameStruct[curframe.id - 1]
                    matches = myVO.featureMatches(curframe.descriptors, prevframe.descriptors)
                    matchedPoints1, matchedPoints2 = myVO.generateMatchedPoints(curframe, prevframe, matches)
                    points, pointIdx = myVO.triangulation(curframe, prevframe, matchedPoints1, matchedPoints2, matches)
       
    fh.close()

    point_cloud = []
    for p in myVO.map.pointCloud:
        point_cloud.append(list(p.point3d))

    structure_point_cloud = []
    for p in myVO.structurePointCloud.mappoint:
        structure_point_cloud.append(p)

    path = []
    for p in myVO.frameStruct:
        path.append(p.t_w.transpose().tolist())

    point_cloud_mat = np.array(point_cloud).transpose()
    structure_point_cloud_mat = np.array(structure_point_cloud).transpose()
    path = np.array(path)
    print(path.shape)
    
    
    np.save("./data/capsule_point_cloud_1_11.npy", point_cloud_mat)
    sio.savemat("./data/capsule_point_cloud_1_11.mat", {'data':point_cloud_mat})

    sio.savemat("./data/capsule_structure_cloud_1_11.mat", {'data':structure_point_cloud_mat})
    sio.savemat("./data/path_1_11.mat", {'data':path})
    






