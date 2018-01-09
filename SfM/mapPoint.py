class MapPoint(object):

    def __init__(self, id = None, point3d = None, viewedframeid = None, descriptors = None):
        if id is not None:
            self.id = id
        else:
            self.id = None

        if point3d is not None:
            self.point3d = point3d
        else:
            self.point3d = None

        if viewedframeid is not None:
            self.viewedframeid = viewedframeid
        else:
            self.viewedframeid = list([])

        if descriptors is not None:
            self.descriptors = descriptors
        else:
            self.descriptors = None

    def updateMapPoint(self, newViewedId = None, point3d = None):
        if newViewedId is not None:
            self.viewedframeid.append(newViewedId)

        if point3d is not None:
            self.point3d = point3d
            print("[MapPoint] Update the point position No. ", self.id)



