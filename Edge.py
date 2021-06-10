import CONSTANTS


class Edge:
    def __init__(self, nodeID1, nodeID2, weight, role, isShortcut=False, middleID=-1):
        '''
        :param nodeID1: int
        :param nodeID2: int
        :param weight: float
        :param role: CONSTANTS.SRC or CONSTANTS.DEST
        :param isShortcut: boolean
        '''
        self.nodeID1 = nodeID1
        self.nodeID2 = nodeID2
        self.weight = weight
        self.role = role
        self.isShortcut = isShortcut
        self.middleID = middleID

    def printInfo(self):
        print("Edge - Node 1: ",    self.nodeID1,
              ", Node 2: ",         self.nodeID2,
              ", with Weight: ",    self.weight,
              ", acting as: ",      self.role
              )