from visualOdometry import VO
from map import Map
import cv2
import os
import numpy as np
from config_default import file_dir
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

associate_file = file_dir['data_dir_tube']
try:
    fh = open(associate_file, 'r')
    assert os.path.getsize(associate_file)
except IOError:
    print("[Error] Failed to open the data_dir")
except AssertionError:
    print("[Warning] There is no data dir in", associate_file, "file!\n")


if __name__ == '__main__':

    map = Map()
    myVO = VO(map)

    ref_t_w = np.array([[0.], [0.], [0.]])
    frame_count = 0
    point_start = 0
    point_end = 0

    for line in fh.readlines():
        '''delete the front blank'''
        line = line.strip()
        image = cv2.imread(line)

        """
        VO.addframe function will proceed:
        (1)feature detection
        (2)add frame to VO.frameStruct, frame id is the length of frameStruct
        """
        curframe = myVO.addframe(image)

        if len(myVO.frameStruct) < 2:
            '''frame 0 with pose r_w(0., 0., 0.) and t_w(0., 0., 0.)'''
            continue

        else:
            if len(myVO.frameStruct) == 2:
                '''frame1 proceed the initialization'''
                myVO.initilization(myVO.frameStruct[0], myVO.frameStruct[1])
                start_count = 0
                end_count = len(myVO.map.pointCloud)

            else:
                '''a new frame added, first to update the pose'''
                try:
                    assert myVO.newframePnP(curframe)
                except AssertionError:
                    print("[Warning] Failed to solve the PnP for current frame, frame id : ", curframe.id)
                    cv2.waitKey()


                '''update the frameStructure'''
                myVO.updateframe(curframe)

                '''update point cloud'''
                prevframe = myVO.frameStruct[curframe.id -1]


                
                if np.linalg.norm(curframe.t_w - ref_t_w) > 5e-1 and np.linalg.norm(curframe.t_w - ref_t_w) < 20:

                    '''update reference position'''
                    ref_t_w = prevframe.t_w
                    #print('ref_t_w', ref_t_w)

                    '''(1) feature matching with previous frame'''
                    matches = myVO.featureMatches(prevframe.descriptors, curframe.descriptors)
                    matchedPoints1, matchedPoints2 = myVO.generateMatchedPoints( prevframe, curframe, matches)
                    '''(2)triangulation'''
                    points, pointIdx = myVO.triangulation(prevframe, curframe, matchedPoints1, matchedPoints2, matches)
                    '''(3)updatePointCloud'''
                    start_count, end_count = myVO.map.updatePointCloud(
                        points, pointIdx, prevframe, curframe, matchedPoints1, matchedPoints2)
                    '''(4)check whether curframe is a keyframe'''
                    if myVO.map.addKeyFrame(curframe, end_count):
                        '''if true, create a keyframe object with curframe inliers'''
                        newly_keypoint_Idx = list(range(start_count, end_count))
                        '''(5)update keyframe'''
                        myVO.map.keyframeset[-1].updatePointList(newly_keypoint_Idx)

                        '''(6)Every time we add a keyframe, we will do the local BA'''
                        '''
                        start_frame = myVO.map.keyframeset[-2].insertframeid
                        end_frame = myVO.map.keyframeset[-1].insertframeid
                        start_point = myVO.map.keyframeset[-2].newpointstart
                        end_point = end_count #the total length of the point cloud
                        #myVO.localBA(start_frame, end_frame, start_point, end_point)
                        '''
                    #myVO.localBA(curframe.id-1, curframe.id, start_count, end_count)

                else:
                    if np.linalg.norm(curframe.t_w - ref_t_w) > 5:
                        curframe.r_w = prevframe.r_w
                        curframe.t_w = prevframe.t_w
                        print("[Warning] one frame error!, frame id: ", curframe.id)


                if curframe.id - frame_count == 50:
                    point_start = point_end
                    point_end = len(myVO.map.pointCloud)
                    print('frame_count:', frame_count, '\n',
                          'point_start', point_start, '\n',
                          'point_end', point_end)

                    myVO.localBA(frame_count, curframe.id, point_start, point_end)

                    frame_count = curframe.id
                    

                print(curframe.t_w)

    #myVO.localBA(0, len(myVO.frameStruct)-1, 0, len(myVO.map.pointCloud))

    fh.close()

    path = []
    for fi in myVO.frameStruct:
        #if np.linalg.norm(fi.t_w) < 30:
        path.append(fi.t_w.reshape(1, 3)[0])
    path = np.array(path).transpose()
    print(path.shape)
    print(path.transpose())

    '''
    points_index = []
    for kf in myVO.map.keyframeset:
        for pi in kf.pointList:
            points_index.append(pi)

    points = []
    points_index = list(set(points_index))
    for pi in points_index:
        #if myVO.map.pointCloud[pi].point3d[0] < 50 and myVO.map.pointCloud[pi].point3d[1] < 50:
        points.append(myVO.map.pointCloud[pi].point3d)
        
    '''

    points = []
    count = 0
    for pi in myVO.map.pointCloud:
        if count % 10 == 0:
            points.append(pi.point3d)
        count += 1
    points = np.array(points).transpose()

    x, y, z = -path[0], -path[1], -path[2]
    xp, yp, zp = points[0], points[1], points[2]

    fig = plt.figure(1)
    ax = Axes3D(fig)
    ax.scatter(x, y, z, 'r')
    ax.plot(x, y, z, 'b')

    fig1 = plt.figure(2)
    ax1 = Axes3D(fig1)
    ax1.scatter(xp, yp, zp, 'b')
    ax1.plot(x, y, z, 'r')
    plt.show()
