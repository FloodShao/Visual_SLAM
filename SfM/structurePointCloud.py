class StructurePointCloud(object):

    def __init__(self, mappoint = None):

        if mappoint is not None:
            self.mappoint = mappoint.point3d
            self.pointcolor = mappoint.pointcolor
        else:
            self.mappoint = []
            self.pointcolor = []



    def addpoint(self, point3d, pointColor):

        self.mappoint.append(point3d)
        self.pointcolor.append(pointColor)

