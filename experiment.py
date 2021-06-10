import csv, pickle, time
import ContractPoINetwork, CONSTANTS
import greedySearch
import collections
import random
import matplotlib.pyplot as plt
import math
import heapq


def rwwr(g, startingQ, theta, distLimit, timeLimit, verbal=True, complexity=True):

    if complexity:
        edgeCount = 0
        startT = time.time()
        #sp, maxDiv = None, -1
        #tRecord, eRecord = [], []

    if verbal:
        print("===========================================")
        print("Start searching for category: ", ','.join([str(x) for x in theta]), " within distance ", distLimit,
              " within ", timeLimit, " sec limit")

    '''
    starting = []
    
    for nID, val in g.nodes.items():
        # Node  --  pois: [ ('museum', (0, 0, 1, 0, 0, 0)),
        #                   ('park', (0, 0, 0, 1, 0, 0)),
        #                   ('zoo', (0, 0, 0, 1, 0, 0)) ]
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
                    print("Find Everything at Starting Point !!!")

                if complexity:
                    stopT = time.time()

                    if verbal:
                        print("Diversity: ", len(stock), " by using ", stopT - startT, " sec and 0 edge !!!")

                    return [nID], [(stopT - startT, sum(theta))], [(0, sum(theta))]
                else:
                    return [nID], None, None

            if len(stock) > 0:
                if verbal:
                    print("Find a starting point ...... In total ", len(starting), " starting points now")

                starting.append((nID, stock))

                if complexity:
                    if maxDiv < len(stock):
                        maxDiv, sp = len(stock), [nID]
                        stopT = time.time()
                        tRecord.append((stopT - startT, maxDiv))
                        eRecord.append((0, maxDiv))

                        if verbal:
                            print("Found a higher diversity: ", maxDiv, " by using ", stopT - startT,
                                  " sec and 0 edge!!!")
    '''

    startingQ = tuple(startingQ)

    while time.time() - startT <= timeLimit:
        if verbal:  print("Another new round from Random Walk ...")

        #(curNode, curRes) = starting[random.randint(0, len(starting) - 1)]

        curNode = random.choice(startingQ)

        curRes = set()

        if len(g.nodes[curNode].pois) > 0:
            cFlag = 0

            for idx in range(CONSTANTS.CATEGORY_NUM):
                # Desire a category and node has corresponding category
                count = min(g.nodes[curNode].category[idx], theta[idx])

                if count > 0:
                    categoryCount = 0
                    for eachP in g.nodes[curNode].pois:
                        if eachP[1][idx] != 0:
                            curRes.add(eachP)
                            categoryCount += 1

                            if categoryCount >= count:  break

                # Current node has more PoIs then enough
                if count == theta[idx]:  cFlag += 1

            if cFlag == CONSTANTS.CATEGORY_NUM:
                stopT = time.time()

                if verbal:
                    print("Find Everything at Starting Point !!!")

                if complexity:
                    if verbal:
                        print("Diversity: ", len(curRes), " by using ", stopT - startT, " sec and 0 edge !!!")

                    return curNode, [curNode], [(stopT - startT, sum(theta))], [(0, sum(theta))]
                else:
                    return curNode, [curNode], None, None

        curSP, curDist = [curNode], 0

        need = [x for x in theta]

        for eachP in curRes:
            for idx in range(CONSTANTS.CATEGORY_NUM):
                need[idx] = max(0, need[idx] - eachP[1][idx])

        if verbal:  print("Starving for category: ", ','.join([str(x) for x in need]))

        while time.time() - startT <= timeLimit:
            outgoingE = []

            for eachE in g.neighborEdges[curNode]:
                if eachE.role == CONSTANTS.SRC and curDist + eachE.weight <= distLimit:
                    outgoingE.append(eachE)

            if len(outgoingE) == 0:
                break

            nextE = outgoingE[random.randint(0, len(outgoingE)-1)]
            nextNode = nextE.nodeID2

            if complexity:  edgeCount += 1

            if len(g.nodes[nextNode].pois) > 0:
                for idx in range(CONSTANTS.CATEGORY_NUM):
                    # Current node has desired category
                    count = min(need[idx], g.nodes[nextNode].category[idx])

                    if count > 0:
                        for eachP in g.nodes[nextNode].pois:
                            if (eachP[1][idx] != 0) and (eachP not in curRes):
                                curRes.add(eachP)

                                count -= 1
                                need[idx] -= 1

                                if count <= 0:
                                    break

                if sum(need) == 0:
                    curSP.append(nextNode)

                    if verbal:
                        print("Found solution and terminated earlier !!!")
                        print("->".join(str(x) for x in curSP))
                        print("===========================================")

                    if complexity:
                        stopT = time.time()
                        #tRecord.append((stopT - startT, sum(theta)))
                        #eRecord.append((edgeCount, sum(theta)))
                        return curSP[0], curSP, [(stopT - startT, sum(theta))], (edgeCount, sum(theta))
                    else:
                        return curSP[0], curSP, None, None
                '''
                else:
                    if complexity:
                        if maxDiv < sum(theta) - sum(need):
                            maxDiv, sp = sum(theta) - sum(need), curSP
                            stopT = time.time()
                            tRecord.append((stopT - startT, maxDiv))
                            eRecord.append((edgeCount, maxDiv))

                            if verbal:
                                print("Found a higher diversity: ", maxDiv, " by using ", stopT - startT,
                                      " sec and ", edgeCount, " edges !!!")
                '''

            curNode = nextNode
            curSP.append(nextNode)
            curDist += nextE.weight

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


def multiSourceDijkstra(g, startingQ, theta, distLimit, verbal=True, complexity=True):
    if verbal:
        print("starting running Dijkstra for ", distLimit, "...")

    if complexity:
        edgeCount, startT = 0, time.time()

    dist = [float('inf')] * len(g.nodes)

    pq = []

    maxDiv, maxNode, path = 0, None, {}

    for eachNode in startingQ:
        dist[eachNode] = 0

        stock = set()

        if len(g.nodes[eachNode].pois) > 0:
            cFlag = 0

            for idx in range(CONSTANTS.CATEGORY_NUM):
                # Desire a category and node has corresponding category
                count = min(g.nodes[eachNode].category[idx], theta[idx])

                if count > 0:
                    categoryCount = 0
                    for eachP in g.nodes[eachNode].pois:
                        if eachP[1][idx] != 0:
                            stock.add(eachP)
                            categoryCount += 1

                            if categoryCount >= count:  break

                # Current node has more PoIs then enough
                if count == theta[idx]:  cFlag += 1

            if cFlag == CONSTANTS.CATEGORY_NUM:
                stopT = time.time()

                if verbal:
                    print("Find Everything at Starting Point !!!")

                if complexity:
                    if verbal:
                        print("Diversity: ", len(stock), " by using ", stopT - startT, " sec and 0 edge !!!")

                    return eachNode, [eachNode], [(stopT - startT, sum(theta))], [(0, sum(theta))]
                else:
                    return eachNode, [eachNode], None, None

        heapq.heappush(pq, (0, eachNode, stock))

        path[eachNode] = None

    visited = set()

    while pq:
        curDist, curNode, curRes = heapq.heappop(pq)

        if curNode in visited:
            continue

        if curDist > distLimit:
            if verbal:  print("Exceed Distance Limit...")

            resHead, resPath = curNode, []

            while resHead is not None:
                resPath.append(resHead)
                resHead = path[resHead]

            resPath.reverse()
            stopT = time.time()

            if complexity:
                return resPath[0], resPath, [(stopT-startT, maxDiv)], [(edgeCount, maxDiv)]
            else:
                return resPath[0], resPath, None, None


        ### Check if find all categories
        need = [x for x in theta]

        for eachP in curRes:
            maxIdx = eachP[1].index(1)
            if need[maxIdx] > 0:
                need[maxIdx] -= 1

        terminateFlag = True

        for x in need:
            if x > 0:
                terminateFlag = False
                break

        if terminateFlag:
            if verbal:  print("Found the results...")

            resHead, resPath = curNode, []

            while resHead is not None:
                resPath.append(resHead)
                resHead = path[resHead]

            resPath.reverse()
            stopT = time.time()

            if complexity:
                return resPath[0], resPath, [(stopT-startT, sum(theta))], [(edgeCount,  sum(theta))]
            else:
                return resPath[0], resPath, None, None

        visited.add(curNode)

        if sum(theta) - sum(need) > maxDiv:
            maxDiv, maxNode = sum(theta) - sum(need), curNode

        for eachE in g.neighborEdges[curNode]:

            if eachE.role == CONSTANTS.DEST:
                continue

            if curDist + eachE.weight > distLimit:
                continue

            if complexity:  edgeCount += 1

            nextNode, nextDist = eachE.nodeID2, curDist + eachE.weight

            if nextDist < dist[nextNode]:
                path[nextNode] = curNode
                dist[nextNode] = nextDist

                nextRes = set([x for x in curRes])

                for eachP in g.nodes[nextNode].pois:
                    nextRes.add(eachP)

                heapq.heappush(pq, (nextDist, nextNode, nextRes))

    resHead, resPath = maxNode, []

    while resHead is not None:
        resPath.append(resHead)
        resHead = path[resHead]

    resPath.reverse()
    stopT = time.time()

    if complexity:
        if verbal:
            print("No solution existed......")
            # if sp is not None:
            #    print("The best diversity: ", maxDiv)
            #    sp.printPath()
            # else:
            #    print("None PoI is in range......")
            print("===========================================")
        return resPath[0], resPath, [(stopT - startT, maxDiv)], [(edgeCount, maxDiv)]
    else:
        return resPath[0], resPath, None, None


def visualRes(greedyRes=None, rwsRes=None):
    if greedyRes:
        # x coordinate, like time or # of edges
        xAxis = set()

        for k, vList in greedyRes.items():
            for v in vList:
                xAxis.add(v[0])

        xAxis = sorted(list(xAxis))

        # y coordinate, diversity
        yAxis = [0] * len(xAxis)

        for k, vList in greedyRes.items():
            # vList = [(x1, div1), (x2, div2), ...]
            tmp = {}

            for eachV in vList:
                tmp[eachV[0]] = eachV[1]

            vList = [(i, tmp[i]) for i in sorted(tmp.keys())]

            ycur, vIdx = 0, 0

            for xIdx, xVal in enumerate(xAxis):
                if vList[vIdx][0] == xVal:
                    ycur = vList[vIdx][1]
                    yAxis[xIdx] += ycur

                    vIdx += 1

                    if vIdx >= len(vList):
                        xIdx += 1

                        while xIdx < len(xAxis):
                            yAxis[xIdx] += ycur
                            xIdx += 1

                        break
                elif vList[vIdx][0] > xVal:
                    yAxis[xIdx] += ycur

        yAxis = [yVal/len(greedyRes) for yVal in yAxis]

        plt.plot(xAxis, yAxis, label='Greedy')

        #print(yAxis)

    if rwsRes:
        # x coordinate, like time or # of edges
        xAxis = set()

        for k, vList in rwsRes.items():
            for v in vList:
                xAxis.add(v[0])

        xAxis = sorted(list(xAxis))

        # y coordinate, diversity
        yAxis = [0] * len(xAxis)

        for k, vList in rwsRes.items():
            # vList = [(x1, div1), (x2, div2), ...]
            tmp = {}

            for eachV in vList:
                tmp[eachV[0]] = eachV[1]

            vList = [(i, tmp[i]) for i in sorted(tmp.keys())]

            ycur, vIdx = 0, 0

            for xIdx, xVal in enumerate(xAxis):
                if vList[vIdx][0] == xVal:
                    ycur = vList[vIdx][1]
                    yAxis[xIdx] += ycur

                    vIdx += 1

                    if vIdx >= len(vList):
                        xIdx += 1

                        while xIdx < len(xAxis):
                            yAxis[xIdx] += ycur
                            xIdx += 1

                        break
                elif vList[vIdx][0] > xVal:
                    yAxis[xIdx] += ycur

        yAxis = [yVal/len(rwsRes) for yVal in yAxis]

        plt.plot(xAxis, yAxis, label='Random Walk')

    plt.legend()
    plt.axis([0, 0.1, 0, 7])
    plt.show()


def main():
    print("Start Loading Diversity......")

    poiDivtDict = {}
    '''
    {
        str_PoI: [0, 0, 0, 1, 0, 0],
        ...
    }
    '''

    with open("ldaModel/NYDivVector.csv", 'r') as rf:
        spamreader = csv.reader(rf)

        for eachPoI in spamreader:
            poiName = eachPoI[0]

            divVec = [eachPoI[i] for i in range(1, CONSTANTS.CATEGORY_NUM+1)]

            poiDivtDict[poiName] = [0] * CONSTANTS.CATEGORY_NUM
            maxIdx = divVec.index(max(divVec))
            poiDivtDict[poiName][maxIdx] = 1

    ############################################################################################
    ############################################################################################

    print("Start Pairing Node with PoIs......")

    poiNoDict = {}

    '''
    {
        int_node_No_1: [ (str_PoI_1, [0, 0, 0, 1]), (str_PoI_2, [0, 1, 0, 0]), ... ]
    }
    Note: Only contain the node with PoI embedded
    '''

    startQ = set()

    with open("poiNetwork/NY_ns.csv", 'r') as rf:
        spamreader = csv.reader(rf)
        next(spamreader)

        for eachN in spamreader:
            nID, nLng, nLat, nPoiStr, nStarting = int(eachN[0]), float(eachN[1]), float(eachN[2]), eachN[3], eachN[4]

            if nPoiStr != '':
                nPois = []
                nPoiL = nPoiStr.split('|')
                for eachP in nPoiL:
                    tmpTuple = (eachP, poiDivtDict[eachP])
                    nPois.append(tmpTuple)

                poiNoDict[nID] = nPois

            if nStarting == 'Y':  startQ.add(nID)

    ############################################################################################
    ############################################################################################

    g = ContractPoINetwork.ContractPoINetwork()

    print("Start Inserting Nodes......")
    with open("poiNetwork/NY_CH_ns.csv", 'r') as rf:
        spamreader = csv.reader(rf)
        next(spamreader)

        count = 1

        for eachN in spamreader:
            nID, nLng, nLat, nDep, nOrd = int(eachN[0]), float(eachN[1]), float(eachN[2]), int(eachN[3]), int(eachN[4])

            if nID not in poiNoDict:
                if nID in startQ:
                    g.addNode(id_=nID, lng_=nLng, lat_=nLat, starting_=True)
                else:
                    g.addNode(id_=nID, lng_=nLng, lat_=nLat, starting_=False)
            else:
                if nID in startQ:
                    g.addNode(id_=nID, lng_=nLng, lat_=nLat, pois_=poiNoDict[nID], starting_=True)
                else:
                    g.addNode(id_=nID, lng_=nLng, lat_=nLat, pois_=poiNoDict[nID], starting_=False)
            '''
            :param depth: Integer, Hierarchy Depth
            :param contractOrder: Integer, contract order
            '''

            g.nodes[nID].depth, g.nodes[nID].contractOrder = nDep, nOrd

            print("Inserted ", count, " nodes")
            g.nodes[nID].printInfo()
            print("///////////////////////")
            count += 1

    print("Inserted all ", count - 1, " nodes successfully......")

    ############################################################################################
    ############################################################################################

    print("Starting Inserting Edges......")
    with open("poiNetwork/NY_CH_es.csv", 'r') as rf:
        spamreader = csv.reader(rf)
        next(spamreader)

        count = 1

        for eachE in spamreader:
            eID1, eID2, eW, eIsShort, eMid = int(eachE[0]), int(eachE[1]), float(eachE[2]), eachE[3], int(eachE[4])

            if math.isnan(eW):
                continue

            if eIsShort == 'n':
                g.addEdge(id1=eID1, id2=eID2, weight=eW)
                print("Inserted ", count, " edges")
            else:
                g.addShortcut(id1=eID1, id2=eID2, weight=eW, midID=eMid)
                print("Inserted ", count, " shortcuts")

            ###print("Inserted ", count, " edges/shortcuts")
            g.edges[(eID1, eID2)].printInfo()
            print("///////////////////////")
            count += 1

    print("Inserted all ", count - 1, " edges successfully......")
    print("==================================================")

    g.rtreeBuild()

    with open("CGContainer2.pickle", 'rb') as f:
        gc = pickle.load(f)

    #print(list(g.tree.intersection((-74.03081879259469, 40.69, -73.97994842847271, 40.72), objects="raw")))

    #st, p, tr, er = greedySearch.greedyProcessStartP(g, gc, [1, 1, 2, 0, 1, 1], 4000, startQ, verbal=False)
    #print(tr)
    #st, p, tr, er = greedySearch.greedyProcessStartP(g, gc, [0, 2, 2, 0, 1, 1], 4000, startQ, verbal=False)
    #print(tr)
    #st, p, tr, er = greedySearch.greedyProcessStartP(g, gc, [0, 1, 2, 1, 1, 1], 5000, startQ, verbal=True)
    #print(tr)
    #exit()

    #st, p, tr, er = greedySearch.greedyProcessStartQ(g, gc, [3, 1, 2, 0, 0, 1], 4000, startQ, verbal=True)
    #print(tr)
    #exit()
    #st, p, tr, er = greedySearch.greedyProcessStartQ(g, gc, [1, 1, 1, 0, 1, 1], 4000, startQ, verbal=False)
    #print(tr)
    #st, p, tr, er = greedySearch.greedyProcessStartQ(g, gc, [0, 1, 2, 1, 1, 1], 5000, startQ, verbal=True)
    #print(tr)
    #greedySearch.greedyProcessWithoutQ(g, gc, [1, 1, 1, 0, 1, 1], 3000, verbal=True)

    sampleNum = 60

    # Generate sample starting points
    sampleStart = False

    if sampleStart:
        startQ = random.sample(list(startQ), k=sampleNum)

        with open('experimentRes/random' + str(sampleNum) + '.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow([str(x) for x in startQ])

        startQ = set(startQ)

        exit()

    # Read starting points from local file
    startFromFile = False

    if startFromFile:

        startQ = set()

        with open('experimentRes/random' + str(sampleNum) + '.csv', 'r') as f:
            rf = csv.reader(f)

            for row in rf:
                if not row:  break
                startQ = set([int(x) for x in row])

    # Run random walk baseline
    baselineRWR = False

    if baselineRWR:
        tfName, tqName = "experimentRes/timeResultP", "experimentRes/timeResultQ"

        if startFromFile:
            tfName += str(sampleNum)
            tqName += str(sampleNum)

        with open(tfName + ".pickle", 'rb') as f:
            trP = pickle.load(f)

        with open(tqName + ".pickle", 'rb') as f:
            trQ = pickle.load(f)

        rwsTimeRes, rwsEdgeRes = {}, {}
        rwsTimeResDouble, rwsEdgeResDouble = {}, {}

        count = 1
        print("Starting experiments of Random Walk with Restart......")

        for k, v in trP.items():
            dist, theta = k[-1], [k[i] for i in range(6)]
            timeLimit = max(v[-1][0], trQ[k][-1][0])

            print("Now processing query with distance limit ", dist, " theta ", ','.join([str(x) for x in theta]),
                  " time limit ", timeLimit)

            spStart, spPath, tRes, eRes = \
                rwwr(g, startQ, theta, distLimit=dist, timeLimit=timeLimit, verbal=False, complexity=True)

            rwsTimeRes[k] = tRes
            rwsEdgeRes[k] = eRes

            print("Now processing query with distance limit ", dist, " theta ", ','.join([str(x) for x in theta]),
                  " double time limit ", 3*timeLimit)

            spStart, spPath, tRes, eRes = \
                rwwr(g, startQ, theta, distLimit=dist, timeLimit=3*timeLimit, verbal=False, complexity=True)

            rwsTimeResDouble[k] = tRes
            rwsEdgeResDouble[k] = eRes

            print("Done ", count, "/", len(trP))
            print("==================================================")

            tfName, efName, tfNameDou, efNameDou = 'experimentRes/timeResultRWS', \
                                                   'experimentRes/edgeResultRWS', \
                                                   'experimentRes/doubleTimeResultRWS', \
                                                   'experimentRes/doubleEdgeResultRWS'

            if startFromFile:
                tfName, efName, tfNameDou, efNameDou = tfName + str(sampleNum), efName + str(sampleNum), \
                                                       tfNameDou + str(sampleNum), efNameDou + str(sampleNum)

            with open(tfName + '.pickle', 'wb') as f:
                pickle.dump(rwsTimeRes, f, pickle.HIGHEST_PROTOCOL)

            with open(efName + '.pickle', 'wb') as f:
                pickle.dump(rwsEdgeRes, f, pickle.HIGHEST_PROTOCOL)

            with open(tfNameDou + '.pickle', 'wb') as f:
                pickle.dump(rwsTimeResDouble, f, pickle.HIGHEST_PROTOCOL)

            with open(efNameDou + '.pickle', 'wb') as f:
                pickle.dump(rwsEdgeResDouble, f, pickle.HIGHEST_PROTOCOL)

            count += 1

        exit()

    # Run Dijkstra baseline
    baselineDij = False

    if baselineDij:
        with open('experimentRes/timeResultP.pickle', 'rb') as f:
            trP = pickle.load(f)

        dijTimeRes, dijEdgeRes = collections.defaultdict(list), collections.defaultdict(list)

        count = 1
        print("Starting experiments of Dijkstra......")

        for k, v in trP.items():
            dist, theta = k[-1], [k[i] for i in range(6)]
            print("Distance: ", k, " theta:", v)

            spStart, spPath, tRes, eRes = multiSourceDijkstra(g, startQ, theta, dist, verbal=False, complexity=True)

            dijTimeRes[(dist, sum(theta))].append(max(0, tRes[-1][1]))
            dijEdgeRes[(dist, sum(theta))].append(max(0, tRes[-1][1]))

            count += 1

            print("Done ", count, "/", len(trP))
            print("==================================================")

        for k, v in dijTimeRes.items():
            print("Avg. diversity of distance limit", k[0], " theta ", k[1], ": ", sum(v)/len(v))

        exit()

    thetaList = []

    with open('experimentRes/randomVar.csv', 'r') as f:
        reader = csv.reader(f)

        for eachTheta in reader:
            thetaList.append([int(x) for x in eachTheta])

    timeResultP, edgeResultP, timeResultQ, edgeResultQ = {}, {}, {}, {}

    distLimit = [500, 700, 1000, 1500, 2500, 3500]

    count = 0

    for theta in thetaList:
        for dist in distLimit:
            print("=======================")
            print("Now processing ", theta, " with distance ", dist)
            print(count+1, "/", len(thetaList)*len(distLimit))

            print("Starting from PoI")
            stP, pP, trP, erP = \
                greedySearch.greedyProcessStartP(g, gc, theta, dist, startQ, verbal=False, complexity=True)
            print("Done with ", trP[-1])

            print("Starting from Query Location")
            stQ, pQ, trQ, erQ = \
                greedySearch.greedyProcessStartQ(g, gc, theta, dist, startQ, verbal=False, complexity=True)
            print("Done with ", trQ[-1])

            timeResultP[tuple(theta) + (dist,)], edgeResultP[tuple(theta) + (dist,)] = trP, erP
            timeResultQ[tuple(theta) + (dist,)], edgeResultQ[tuple(theta) + (dist,)] = trQ, erQ

            tpfName, epfName = 'experimentRes/timeResultP', 'experimentRes/edgeResultP'
            tqfName, eqfName = 'experimentRes/timeResultQ', 'experimentRes/edgeResultQ'

            if startFromFile:
                tpfName, epfName, tqfName, eqfName = tpfName + str(sampleNum), epfName + str(sampleNum), \
                                                     tqfName + str(sampleNum), eqfName + str(sampleNum)

            with open(tpfName + '.pickle', 'wb') as f:
                pickle.dump(timeResultP, f, pickle.HIGHEST_PROTOCOL)

            with open(epfName + '.pickle', 'wb') as f:
                pickle.dump(edgeResultP, f, pickle.HIGHEST_PROTOCOL)

            with open(tqfName + '.pickle', 'wb') as f:
                pickle.dump(timeResultQ, f, pickle.HIGHEST_PROTOCOL)

            with open(eqfName + '.pickle', 'wb') as f:
                pickle.dump(edgeResultQ, f, pickle.HIGHEST_PROTOCOL)

            count += 1


if __name__ == '__main__':
    main()