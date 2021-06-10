import time
import heapq
from functools import total_ordering
import CONSTANTS
import math
import collections

@total_ordering
class Element:
    def __init__(self, nid, priority=0, dist=0, res=None, parent=None, explored=None):
        '''
        :param nid:  int, ID of node
        :param priority:  float (default: 0), priority, the lower the better
        :param dist:  float (default: 0), distance from nid to start
        :param res:  set (default: {}), collected PoI(s)
        :param parent:  Element (default: None (as starting point)), parent of current Element
        :param explored:  set (default: {}), explored edges
        '''

        if res is None:  res = set()
        if explored is None: explored = set()

        self.nid = nid
        self.priority = priority
        self.dist = dist
        self.res = res
        self.parent = parent
        self.explored = explored

    def __eq__(self, other):
        return (self.priority, self.dist, len(self.res), len(self.explored)) == \
               (other.priority, other.dist, len(other.res), len(other.explored))

    def __lt__(self, other):
        return (self.priority, self.dist, - len(self.res), len(self.explored)) < \
               (other.priority, other.dist, - len(other.res), len(other.explored))

    def printPath(self):
        path, cur = [], self

        while cur:
            path.append(cur.nid)

            cur = cur.parent

        path.reverse()

        #print("//////////////////////////////")
        #print("Path:: ", "->".join([str(x) for x in path]))
        #print("//////////////////////////////")

        return path

    def findHead(self):
        head, cur = None, self

        while cur:
            head = cur.nid
            cur = cur.parent

        return head


def greedyProcessStartP(g, gc, theta, maxDist, startingQ, verbal=True, complexity=True):
    '''
    :param g: PoI network
    :param gc: geometric container
    :param theta: categorical require [1, 2, 0, 2, 3, 1]
    :param maxDist: distance limit
    :param startingQ: set(), Node ID of starting query locations
    '''

    if verbal:
        print("===========================================")
        print("Start searching for category: ", ','.join([str(x) for x in theta]), " within distance ", maxDist)

    if complexity:
        startT, edgeCount = time.time(), 0
        #sp, maxDiv = None, -1
        #tRecord, eRecord = [], []

    pq = []
    queried = collections.defaultdict(lambda : -1)

    for nID, val in g.nodes.items():
        # Node  --  pois: [ ('museum', (0, 0, 1, 0, 0, 0)),
        #                   ('park',   (0, 0, 0, 1, 0, 0)),
        #                   ('zoo',    (0, 0, 0, 1, 0, 0)) ]
        #           category: [0, 0, 1, 2, 0, 0]
        if len(val.pois) > 0:
            stock = set()
            cFlag = 0

            for idx in range(CONSTANTS.CATEGORY_NUM):
                # Desire a category and node has corresponding category
                count = min(val.category[idx], theta[idx])

                if count > 0:
                    categoryCount = 0

                    for eachP in val.pois:
                        if eachP[1][idx] != 0:
                            stock.add(eachP)
                            categoryCount += 1

                            if categoryCount >= count:  break

                # Current node has more PoIs then enough
                if count == theta[idx]:  cFlag += 1

            if cFlag == CONSTANTS.CATEGORY_NUM:
                if verbal:
                    print("Find Everything at a PoI location !!!")

                ## Now find if starting point within range

                queried[nID] = maxDist

                restCost = maxDist
                restCost /= CONSTANTS.REarth

                dLat = restCost
                dLng = restCost/math.cos(CONSTANTS.varPiDegree * val.lat)

                dLat = dLat/CONSTANTS.varPiDegree
                dLng = dLng/CONSTANTS.varPiDegree

                lngMin, latMin, lngMax, latMax = min(val.lng + dLng, val.lng - dLng),  \
                                                 min(val.lat + dLat, val.lat - dLat), \
                                                 max(val.lng + dLng, val.lng - dLng), \
                                                 max(val.lat + dLat, val.lat - dLat)

                rangeRes = list(g.tree.intersection((lngMin, latMin, lngMax, latMax), objects="raw"))

                openSet = set([x for x in rangeRes if x in startingQ])

                if verbal:
                    print("Range query of ", val.lng, val.lat, " with budget ", maxDist)
                    print("i.e., bounding box -- ", lngMin, lngMax, latMin, latMax)
                    print("Found in total ", len(openSet), " candidates...")

                if len(openSet) > 0:
                    start2pPath, start2pDist = g.biDijkstra(openSet, nID, maxDist)

                    if start2pDist <= maxDist:
                        if verbal:
                            print("Path: ", "->".join(str(x) for x in start2pPath))

                        if complexity:
                            stopT = time.time()

                            if verbal:
                                print("Diversity: ", len(stock), " by using ", stopT - startT, " sec and ",
                                      len(start2pPath), " edges !!!")
                                print(stock)

                            ###return Element(nid=nID, res=stock), [(stopT - startT, sum(theta))], [(0, sum(theta))]

                            return start2pPath[0], start2pPath, [(stopT - startT, sum(theta))], [(0, sum(theta))]
                        else:
                            return start2pPath[0], start2pPath, None, None
            else:
                if len(stock) > 0:
                    if verbal:
                        print("Find a starting point ...... In total ", len(pq), " starting points now")

                    heapq.heappush(pq, Element(nid=nID, res=stock))

                ### Comment the following part is because the searching starts from PoI and
                ### it is possible to have no Starting Point around it => contribute nothing to diversity of results
                '''
                if complexity:
                    if maxDiv < len(stock):
                        maxDiv, sp = len(stock), tmp
                        stopT = time.time()
                        tRecord.append((stopT - startT, maxDiv))
                        eRecord.append((0, maxDiv))

                        if verbal:
                            print("Found a higher diversity: ", maxDiv, " by using ", stopT - startT,
                                  " sec and 0 edge!!!")
                '''

    while pq:
        if verbal:  print(len(pq), " elements awaiting in queue...")

        curElement = heapq.heappop(pq)

        #curDiv = [0] * CONSTANTS.CATEGORY_NUM

        # Calculate the diversity of collected PoIs in current node
        #for eachP in curElement.res:
        #    maxIdx = eachP[1].index(1)
        #    curDiv[maxIdx] += 1
        #    for idx in range(CONSTANTS.CATEGORY_NUM):
        #        curDiv[idx] += eachP[1][idx]

        # for idx in range(CONSTANTS.CATEGORY_NUM):
        #    need[idx] = theta[idx] - curDiv[idx]

        need = [x for x in theta]

        for eachP in curElement.res:
            maxIdx = eachP[1].index(1)
            need[maxIdx] -= 1

        if verbal:
            print("Starving for category: ", ','.join([str(x) for x in need]))

        for eachE in g.neighborEdges[curElement.nid]:
            if complexity:  edgeCount += 1

            if eachE.role == CONSTANTS.SRC:
                adj = eachE.nodeID2

                if (curElement.nid, adj) not in curElement.explored:
                    nextMinDist = 0
                    nextMaxDist = curElement.dist + eachE.weight

                    for idx in range(CONSTANTS.CATEGORY_NUM):
                        if need[idx] > 0:
                            nextMinDist = max(nextMinDist, gc[adj][idx])
                            nextMaxDist += 2 * gc[adj][idx]

                    if curElement.dist + eachE.weight + nextMinDist <= maxDist:
                        adjRes = set([x for x in curElement.res])

                        # Add necessary PoIs into .res
                        # If current node does not have any PoI, then ignore
                        if len(g.nodes[adj].pois) > 0:
                            for idx in range(CONSTANTS.CATEGORY_NUM):
                                # Current node has desired category
                                count = min(need[idx], g.nodes[adj].category[idx])

                                if count > 0:
                                    for eachP in g.nodes[adj].pois:
                                        if (eachP[1][idx] != 0) and (eachP not in adjRes):
                                            adjRes.add(eachP)

                                            count -= 1

                                            if count <= 0:
                                                break

                        adjDist = curElement.dist + eachE.weight

                        adjExplored = set([x for x in curElement.explored])

                        adjExplored.add((curElement.nid, adj))

                        #adjN = Element(nid=adj, priority=nextD, dist=adjDist, res=adjRes,
                        #               parent=curElement, explored=adjExplored)

                        # min: adjDist + nextMinDist
                        # max: nextMaxDist - nextMinDist
                        #adjN = Element(nid=adj, priority=0.5*(adjDist + nextMaxDist),
                        #               dist=adjDist, res=adjRes,
                        #               parent=curElement, explored=adjExplored)

                        # min: adjDist + nextMinDist
                        # max: nextMaxDist - nextMinDist
                        # 0.8 * min + 0.2 * max =
                        #   0.8 * adjDist + 0.8 * nextMinDist + 0.2 * nextMaxDist - 0.2 * NextMinDist
                        factorPriority = CONSTANTS.factorPrior ** sum(need)
                        adjN = Element(nid=adj, priority=factorPriority * adjDist +
                                                         (1 - factorPriority) * nextMaxDist +
                                                         (2 * factorPriority - 1) * nextMinDist,
                                       dist=adjDist, res=adjRes, parent=curElement, explored=adjExplored)

                        # Check the diversity to see if we found the results
                        adjDiv = [0] * CONSTANTS.CATEGORY_NUM

                        for eachP in adjRes:
                            maxIdx = eachP[1].index(1)
                            adjDiv[maxIdx] += 1

                        '''
                        if complexity:
                            if maxDiv < sum(adjDiv):
                                maxDiv, sp = sum(adjDiv), adjN
                                stopT = time.time()
                                tRecord.append((stopT - startT, maxDiv))
                                eRecord.append((edgeCount, maxDiv))

                                if verbal:
                                    print("Found a higher diversity: ", maxDiv, " by using ", stopT - startT,
                                          " sec and ", edgeCount, " edges !!!")
                        '''

                        terminateFlag = True

                        for idx in range(CONSTANTS.CATEGORY_NUM):
                            if adjDiv[idx] < theta[idx]:
                                # If our collections is less than the requirement for category, then searching continues
                                terminateFlag = False
                                break

                        if terminateFlag:

                            srcHead = adjN.findHead()

                            ### rtree
                            restCost = maxDist - adjDist

                            if queried[srcHead] >= restCost:  continue

                            queried[srcHead] = restCost

                            restCost /= CONSTANTS.REarth

                            dLat = restCost
                            dLng = restCost / math.cos(CONSTANTS.varPiDegree * g.nodes[srcHead].lat)

                            dLat = dLat/CONSTANTS.varPiDegree
                            dLng = dLng/CONSTANTS.varPiDegree

                            lngMin, latMin, lngMax, latMax = \
                                min(g.nodes[srcHead].lng + dLng, g.nodes[srcHead].lng - dLng), \
                                min(g.nodes[srcHead].lat + dLat, g.nodes[srcHead].lat - dLat), \
                                max(g.nodes[srcHead].lng + dLng, g.nodes[srcHead].lng - dLng), \
                                max(g.nodes[srcHead].lat + dLat, g.nodes[srcHead].lat - dLat)

                            rangeRes = list(g.tree.intersection((lngMin, latMin, lngMax, latMax), objects="raw"))

                            openSet = set([x for x in rangeRes if x in startingQ])

                            if verbal:
                                print("Range query of ", g.nodes[srcHead].lng, g.nodes[srcHead].lat,
                                      " with budget ", maxDist - adjDist)
                                print("i.e., bounding box -- ", lngMin, lngMax, latMin, latMax)
                                print("Found in total ", len(openSet), " candidates...")

                            if len(openSet) > 0:
                                start2pPath, start2pDist = g.biDijkstra(openSet, srcHead, maxDist - adjDist)

                                if start2pDist <= maxDist - adjDist:
                                    if verbal:
                                        print("Found solution and terminated earlier !!! AT apex ", srcHead)
                                        finalPath = start2pPath[:-1] + adjN.printPath()
                                        print("Path: ", "->".join(str(x) for x in finalPath))
                                        print(adjRes)
                                        print("===========================================")

                                    if complexity:
                                        #maxDiv, sp = sum(adjDiv), adjN
                                        stopT = time.time()
                                        #tRecord.append((stopT - startT, maxDiv))
                                        #eRecord.append((edgeCount, maxDiv))

                                        return start2pPath[0], start2pPath[:-1] + adjN.printPath(), \
                                               [(stopT - startT, sum(theta))], [(edgeCount, sum(theta))]
                                    else:
                                        return start2pPath[0], start2pPath[:-1] + adjN.printPath(), None, None
                        else:
                            heapq.heappush(pq, adjN)

    if complexity:
        if verbal:
            print("No solution existed......")
            #if sp is not None:
            #    print("The best diversity: ", maxDiv)
            #    sp.printPath()
            #else:
            #    print("None PoI is in range......")
            print("===========================================")
        stopT = time.time()
        return None, [], [(stopT - startT, -1)], [(edgeCount, -1)]
    else:
        return None, [], None, None



def greedyProcessStartQ(g, gc, theta, maxDist, startingQ, verbal=True, complexity=True):
    '''
    :param g: PoI network
    :param gc: geometric container
    :param theta: categorical require [1, 2, 0, 2, 3, 1]
    :param maxDist: distance limit
    :param startingQ: set(), Node ID of starting query locations
    '''

    if verbal:
        print("===========================================")
        print("......Starting Query Location Version......")
        print("Start searching for category: ", ','.join([str(x) for x in theta]), " within distance ", maxDist)
        print("Number of starting points: ", len(startingQ))

    if complexity:
        startT, edgeCount = time.time(), 0
        #sp, maxDiv = None, -1
        #tRecord, eRecord = [], []

    pq = []

    for nID in startingQ:
        # Node  --  pois: [ ('museum', (0, 0, 1, 0, 0, 0)),
        #                   ('park',   (0, 0, 0, 1, 0, 0)),
        #                   ('zoo',    (0, 0, 0, 1, 0, 0)) ]
        #           category: [0, 0, 1, 2, 0, 0]
        stock = set()

        # If there is some PoIs in query location
        if len(g.nodes[nID].pois) > 0:
            cFlag = 0

            for idx in range(CONSTANTS.CATEGORY_NUM):
                # Desire a category and node has corresponding category
                count = min(g.nodes[nID].category[idx], theta[idx])

                if count > 0:
                    categoryCount = 0
                    for eachP in g.nodes[nID].pois:
                        if eachP[1][idx] != 0:
                            stock.add(eachP)
                            categoryCount += 1

                            if categoryCount >= count:  break

                # Current node has more PoIs then enough
                if count == theta[idx]:  cFlag += 1

            if cFlag == CONSTANTS.CATEGORY_NUM:
                if verbal:
                    print("Find Everything at Starting Point !!!")

                if complexity:
                    stopT = time.time()

                    if verbal:
                        print("Diversity: ", len(stock), " by using ", stopT - startT, " sec and 0 edge !!!")

                    return nID, [nID], [(stopT - startT, sum(theta))], [(0, sum(theta))]
                else:
                    return nID, [nID], None, None

        heapq.heappush(pq, Element(nid=nID, res=stock))

        '''
        if complexity:
            if maxDiv < len(stock):
                maxDiv, sp = len(stock), tmp
                stopT = time.time()
                tRecord.append((stopT - startT, maxDiv))
                eRecord.append((0, maxDiv))

            if verbal:
                print("Found a higher diversity: ", maxDiv, " by using ", stopT - startT, " sec and 0 edge!!!")
        '''

    while pq:
        if verbal:  print(len(pq), " elements awaiting in queue...")

        curElement = heapq.heappop(pq)

        #curDiv = [0] * CONSTANTS.CATEGORY_NUM

        # Calculate the diversity of collected PoIs in current node
        #for eachP in curElement.res:
        #    for idx in range(CONSTANTS.CATEGORY_NUM):
        #        curDiv[idx] += eachP[1][idx]

        #need = [0] * CONSTANTS.CATEGORY_NUM

        #for idx in range(CONSTANTS.CATEGORY_NUM):
        #    need[idx] = theta[idx] - curDiv[idx]

        need = [x for x in theta]

        for eachP in curElement.res:
            maxIdx = eachP[1].index(1)
            need[maxIdx] -= 1

        if verbal:
            print("Starving for category: ", ','.join([str(x) for x in need]))

        for eachE in g.neighborEdges[curElement.nid]:
            if complexity:  edgeCount += 1

            if eachE.role == CONSTANTS.SRC:
                adj = eachE.nodeID2

                if (curElement.nid, adj) not in curElement.explored:
                    nextMinDist = 0
                    nextMaxDist = curElement.dist + eachE.weight

                    for idx in range(CONSTANTS.CATEGORY_NUM):
                        if need[idx] > 0:
                            nextMinDist = max(nextMinDist, gc[adj][idx])
                            nextMaxDist += 2 * gc[adj][idx]

                    if curElement.dist + eachE.weight + nextMinDist <= maxDist:
                        adjRes = set([x for x in curElement.res])

                        # Add necessary PoIs into .res
                        # If current node does not have any PoI, then ignore
                        if len(g.nodes[adj].pois) > 0:
                            for idx in range(CONSTANTS.CATEGORY_NUM):
                                # Current node has desired category
                                count = min(need[idx], g.nodes[adj].category[idx])

                                if count > 0:
                                    for eachP in g.nodes[adj].pois:
                                        if (eachP[1][idx] != 0) and (eachP not in adjRes):
                                            adjRes.add(eachP)

                                            count -= 1

                                            if count <= 0:
                                                break

                        adjDist = curElement.dist + eachE.weight

                        adjExplored = set([x for x in curElement.explored])

                        adjExplored.add((curElement.nid, adj))

                        #adjN = Element(nid=adj, priority=nextD, dist=adjDist, res=adjRes,
                        #               parent=curElement, explored=adjExplored)

                        # Priority: min: adjDist + nextMinDist
                        #           max: nextMaxDist - nextMinDist
                        #adjN = Element(nid=adj, priority=0.5*(adjDist + nextMaxDist),
                        #               dist=adjDist, res=adjRes,
                        #               parent=curElement, explored=adjExplored)

                        # min: adjDist + nextMinDist
                        # max: nextMaxDist - nextMinDist
                        # 0.8 * min + 0.2 * max =
                        #   0.8 * adjDist + 0.8 * nextMinDist + 0.2 * nextMaxDist - 0.2 * NextMinDist
                        factorPriority = CONSTANTS.factorPrior ** sum(need)
                        adjN = Element(nid=adj, priority=factorPriority * adjDist +
                                                         (1-factorPriority) * nextMaxDist +
                                                         (2*factorPriority - 1) * nextMinDist,
                                       dist=adjDist, res=adjRes, parent=curElement, explored=adjExplored)

                        #print("Path: ", "->".join([str(x) for x in adjN.printPath()]))

                        # Check the diversity to see if we found the results
                        adjDiv = [0] * CONSTANTS.CATEGORY_NUM

                        for eachP in adjRes:
                            maxIdx = eachP[1].index(1)
                            adjDiv[maxIdx] += 1

                        '''
                        if complexity:
                            if maxDiv < sum(adjDiv):
                                maxDiv, sp = sum(adjDiv), adjN
                                stopT = time.time()
                                tRecord.append((stopT - startT, maxDiv))
                                eRecord.append((edgeCount, maxDiv))

                                if verbal:
                                    print("Found a higher diversity: ", maxDiv, " by using ", stopT - startT,
                                          " sec and ", edgeCount, " edges !!!")
                        '''

                        terminateFlag = True

                        for idx in range(CONSTANTS.CATEGORY_NUM):
                            if adjDiv[idx] < theta[idx]:
                                # If our collections is less than the requirement for category, then searching continues
                                terminateFlag = False
                                break

                        if terminateFlag:
                            if verbal:
                                print("Found solution and terminated earlier !!!")
                                print("Path: ", "->".join([str(x) for x in adjN.printPath()]))
                                print(adjRes)
                                print("===========================================")

                            if complexity:
                                stopT = time.time()
                                return adjN.findHead(), adjN.printPath(), \
                                       [(stopT - startT, sum(theta))], [(edgeCount, sum(theta))]
                                #return adjN.findHead(), adjN.printPath(), tRecord, eRecord
                            else:
                                return adjN.findHead(), adjN.printPath(), None, None

                        heapq.heappush(pq, adjN)

    if complexity:
        if verbal:
            print("No solution existed......")
            # if sp is not None:
            #    print("The best diversity: ", maxDiv)
            #    sp.printPath()
            # else:
            #    print("None PoI is in range......")
            print("===========================================")
        stopT = time.time()
        return None, [], [(stopT - startT, -1)], [(edgeCount, -1)]
    else:
        return None, [], None, None