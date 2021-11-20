import CONSTANTS


class Edge:
    def __init__(self, node_id1, node_id2, weight, role, isShortcut=False, middle_id=-1):
        """
        :param node_id1: integer
        :param node_id2: integer
        :param weight: float
        :param role: CONSTANTS.SRC or CONSTANTS.DEST
        :param isShortcut: boolean
        """
        self.node_id1 = node_id1
        self.node_id2 = node_id2
        self.weight = weight
        self.role = role
        self.isShortcut = isShortcut
        self.middle_id = middle_id

    def print_info(self):
        print("Edge - Node 1: ", self.node_id1,
              ", Node 2: ", self.node_id2,
              ", with Weight: ", self.weight,
              ", acting as: ", self.role,
              ", is shortcut: ", self.isShortcut)
