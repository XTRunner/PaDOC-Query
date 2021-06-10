import ContractPoINetwork
import CONSTANTS

import csv, time
import multiprocessing as mp

from itertools import repeat


def main():
    poiDivtDict = {}

    with open("ldaModel_12/NYDivVector.csv", 'r') as rf:
        spamreader = csv.reader(rf)

        for eachPoI in spamreader:
            poiName = eachPoI[0]

            divVec = [eachPoI[i] for i in range(1, CONSTANTS.CATEGORY_NUM+1)]

            poiDivtDict[poiName] = [0] * CONSTANTS.CATEGORY_NUM
            maxIdx = divVec.index(max(divVec))
            poiDivtDict[poiName][maxIdx] = 1

    g = ContractPoINetwork.ContractPoINetwork()

    print("Start Inserting Nodes......")
    with open("poiNetwork/NY_ns.csv", 'r') as rf:
        spamreader = csv.reader(rf)
        next(spamreader)

        count = 1

        for eachN in spamreader:
            nID, nLng, nLat, nPoiStr = int(eachN[0]), float(eachN[1]), float(eachN[2]), eachN[3]

            if eachN[4] == 'Y':
                nStarting = True
            else:
                nStarting = False

            if nPoiStr == '':
                g.addNode(id_=nID, lng_=nLng, lat_=nLat, starting_=nStarting)
            else:
                nPois = []
                nPoiL = nPoiStr.split('|')

                for eachP in nPoiL:
                    tmpTuple = (eachP, poiDivtDict[eachP])
                    nPois.append(tmpTuple)

                g.addNode(id_=nID, lng_=nLng, lat_=nLat, pois_=nPois, starting_=nStarting)

            print("Inserted ", count, " nodes")
            g.nodes[nID].printInfo()
            print("///////////////////////")
            count += 1

    print("Inserted all ", count - 1, " nodes successfully......")

    print("Starting Inserting Edges......")
    with open("poiNetwork/NY_es.csv", 'r') as rf:
        spamreader = csv.reader(rf)
        next(spamreader)

        count = 1

        for eachE in spamreader:
            eID1, eID2, eW = int(eachE[0]), int(eachE[1]), float(eachE[2])
            g.addEdge(id1=eID1, id2=eID2, weight=eW)
            print("Inserted ", count, " edges")
            g.edges[(eID1, eID2)].printInfo()
            print("///////////////////////")
            count += 1

    print("Inserted all ", count - 1, " edges successfully......")

    print("Start contracting network with ", mp.cpu_count(), " CPUs")

    startT = time.time()

    while True:

        uncontractedNodes = [nID for nID, nNode in g.nodes.items() if not nNode.isContracted]

        if len(uncontractedNodes) == 0:
            break

        print("=========================")
        print(len(uncontractedNodes), " nodes have not been contracted......")

        if g.curIter != 0:  # Only update the neighbors of the contracted node in the last iteration
            uncontractedNodes = set()
            for eachContractedN in independentSet:
                adjEdges = g.neighborEdges[eachContractedN]
                for eachE in adjEdges:
                    if (eachE.nodeID1 != eachContractedN) and (not g.nodes[eachE.nodeID1].isContracted):
                        uncontractedNodes.add(eachE.nodeID1)
                    if (eachE.nodeID2 != eachContractedN) and (not g.nodes[eachE.nodeID2].isContracted):
                        uncontractedNodes.add(eachE.nodeID2)

        print(len(uncontractedNodes), " nodes have to update priority......")
        print("=========================")

        pool = mp.Pool(mp.cpu_count())

        priorityRes = pool.map(g.priorityCal, [i for i in uncontractedNodes])

        pool.close()

        #g.shortcuts.clear()

        for pID, pPriority, pShortcuts in priorityRes:
            g.nodes[pID].priority = pPriority  # Update Priority
            g.shortcuts[pID] = pShortcuts  # Update shortcuts

        ### Start find Independent set ###

        uncontractedNodes = [nID for nID, nNode in g.nodes.items() if not nNode.isContracted]

        managerIndependent = mp.Manager()

        independentDone = managerIndependent.dict()

        for i in uncontractedNodes:
            independentDone[i] = True

        pool = mp.Pool(mp.cpu_count())

        independentRes = pool.starmap(g.isIndependent,
                                      zip(uncontractedNodes, repeat(independentDone))
                                      )

        pool.close()

        independentList = []

        # Pick the CONSTANTS.SizeOfIndependentSet nodes with the smallest priority
        for iID, iIndependent in independentRes:
            if iIndependent:  # If independent, return True
                independentList.append((iID, g.nodes[iID].priority))

        '''
        sortedIndependentList = sorted(independentList, key=lambda x: x[1])

        if sortedIndependentList[0][1] == 1.0:  # First iteration - if priority == 1.0, no shortcut
            independentSet = [x[0] for x in sortedIndependentList if x[1] == 1.0]
        else:
            independentSet = [val[0] for idx, val in enumerate(sortedIndependentList)
                              if idx < CONSTANTS.SizeOfIndependentSet]
        '''

        if g.curIter == 0:  # First iteration => if priority == 1.0 and thus no shortcut/neighbor, dead-end node
            independentSet = [x[0] for x in independentList if x[1] == 1.0]
        else:
            if len(independentList) > CONSTANTS.SizeOfIndependentSet:
                independentList = sorted(independentList, key=lambda x: x[1])

            independentSet = [val[0] for idx, val in enumerate(independentList)
                              if idx < CONSTANTS.SizeOfIndependentSet]

        print("=========================")
        print(len(independentSet), " independent nodes are going to be contracted......")
        print("=========================")

        # s1: Add shortcuts to self.edges
        # s2: Add shortcuts to self.neighbor
        # s3: Update depth of neighbors for each contracting node
        # s4: Mark node.isContracted as True and Update contractOrder to g.curIter
        # s5: Update g.curIter

        for contractingNode in independentSet:
            # s1, s2
            shortcutList = g.shortcuts[contractingNode]

            for scCost, scSrcDest in shortcutList:
                # (scCost, (scSrc, scDest)) src -> contractingNode -> dest
                scSrc, scDest = scSrcDest
                g.addShortcut(scSrc, scDest, scCost, contractingNode)

            # s3
            g.updateDepth(contractingNode)

            # s4
            g.nodes[contractingNode].isContracted = True
            g.nodes[contractingNode].contractOrder = g.curIter

            # s5
            g.curIter += 1

        print("//////////////////////////////////////////////")

    endT = time.time()

    print("Time Cost: ", endT-startT)

    with open('poiNetwork/NY_CH_ns_12.csv', 'w', newline='') as wfile:
        spamwriter = csv.writer(wfile)

        spamwriter.writerow(['no', 'lng', 'lat', 'depth', 'order'])

        for k, v in g.nodes.items():
            spamwriter.writerow([k, v.lng, v.lat, v.depth, v.contractOrder])

    with open('poiNetwork/NY_CH_es_12.csv', 'w', newline='') as wfile:
        spamwriter = csv.writer(wfile)

        spamwriter.writerow(['start', 'end', 'weight', 'isShortcut', 'mid'])

        for k, v in g.edges.items():
            nStart, nEnd, nW, nMid = v.nodeID1, v.nodeID2, v.weight, v.middleID

            if v.isShortcut:
                nSC = 'Y'
            else:
                nSC = 'N'

            spamwriter.writerow([nStart, nEnd, nW, nSC, nMid])


if __name__ == "__main__":
    main()