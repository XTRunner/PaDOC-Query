import CONSTANTS


class PoI:
    def __init__(self, poi_id: int, name: str, category: int):
        """
        :param poi_id: integer, unique ID for each PoI
        :param name: string, name of PoI
        :param category: integer, index from 0 to CONSTANTS.CATEGORY_NUM - 1
        """
        self.poi_id = poi_id
        self.name = name
        self.category = category

    def print_info(self):
        print("PoI: ", self.poi_id,
              ", Name: ", self.name,
              ", belonged to category ", self.category)


class Node:
    def __init__(self, node_id: int, lng: float, lat: float, pois=None, priority=0, isContracted=False, depth=1):
        """
        :param node_id: integer
        :param lng: float
        :param lat: float
        :param pois: set(class PoI)
        :param priority: float
        :param isContracted: boolean, True of contracted already, otherwise False
        :param depth: integer, Hierarchy Depth
        """
        self.node_id = node_id
        self.lng = lng
        self.lat = lat

        self.PoIs = set()

        if pois is not None:
            for each_poi in pois:
                self.PoIs.add(each_poi)

        self.category = [0] * CONSTANTS.CATEGORY_NUM

        if len(self.PoIs) > 0:
            for each_poi in self.PoIs:
                self.category[each_poi.category] += 1

        self.priority = priority
        self.isContracted = isContracted
        self.depth = depth
        self.contract_order = -1

    def print_info(self):
        print("Node with ID: ", self.node_id,
              ", lng: ", self.lng,
              ", lat: ", self.lat,
              ", category: ", self.category,
              ", priority: ", self.priority,
              ", contracted: ", self.isContracted,
              ", depth: ", self.depth)
