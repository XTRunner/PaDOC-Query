import CONSTANTS


class Node:
    def __init__(self, nodeID, lng, lat, pois=None, priority=0, isContracted=False, depth=1, starting=False):
        '''
        :param nodeID: integer
        :param lng: float
        :param lat: float
        :param pois: [(poiName1, [...]), (poiName2, [......])]
        :param priority: float
        :param isContracted: boolean, True of contracted already, otherwise False
        :param depth: Integer, Hierarchy Depth
        :param starting: boolean, True of starting point, otherwise False
        '''
        self.nodeID = nodeID
        self.lng = lng
        self.lat = lat

        #if pois is None:
        #    self.pois = []
        #else:
        #    self.pois = pois

        self.pois = []

        if pois is not None:
            for eachP in pois:
                self.pois.append((eachP[0], tuple(eachP[1])))

        self.category = [0] * CONSTANTS.CATEGORY_NUM

        if len(self.pois) > 0:
            for eachPoI in self.pois:
                categoryIdx = eachPoI[1].index(1)
                self.category[categoryIdx] += 1

        self.priority = priority
        self.isContracted = isContracted
        self.depth = depth
        self.contractOrder = -1
        self.starting = starting

    def printInfo(self):
        print("Node with ID: ", self.nodeID,
              ", lng: ",        self.lng,
              ", lat: ",        self.lat,
              ", PoIs ",        self.pois,
              ", category: ",   self.category,
              ", priority: ",   self.priority,
              ", contracted: ", self.isContracted,
              ", depth: ",      self.depth
              )