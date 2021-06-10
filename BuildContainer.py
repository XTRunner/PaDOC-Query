import pickle
import multiprocessing as mp
import csv, math
import ContractPoINetwork, CONSTANTS


def main():

    print("Start Loading Diversity......")

    poiDivtDict = {}
    '''
    {
        str_PoI: [0, 0, 0, 1, 0, 0],
        ...
    }
    '''

    with open("ldaModel_12/NYDivVector.csv", 'r') as rf:
        spamreader = csv.reader(rf)

        for eachPoI in spamreader:
            poiName = eachPoI[0]

            divVec = [eachPoI[i] for i in range(1, CONSTANTS.CATEGORY_NUM+1)]

            poiDivtDict[poiName] = [0] * CONSTANTS.CATEGORY_NUM
            maxIdx = divVec.index(max(divVec))
            poiDivtDict[poiName][maxIdx] = 1

            print("PoI: ", poiName, " belonged to category ", maxIdx)

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
    startingPs = set()

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

            if nStarting == 'Y':  startingPs.add(nID)

    ############################################################################################
    ############################################################################################

    g = ContractPoINetwork.ContractPoINetwork()

    print("Start Inserting Nodes......")
    with open("poiNetwork/NY_CH_ns_12.csv", 'r') as rf:
        spamreader = csv.reader(rf)
        next(spamreader)

        count = 1

        for eachN in spamreader:
            nID, nLng, nLat, nDep, nOrd = int(eachN[0]), float(eachN[1]), float(eachN[2]), int(eachN[3]), int(eachN[4])

            if nID in startingPs:
                nStarting = True
            else:
                nStarting = False

            if nID not in poiNoDict:
                g.addNode(id_=nID, lng_=nLng, lat_=nLat, starting_=nStarting)
            else:
                g.addNode(id_=nID, lng_=nLng, lat_=nLat, pois_=poiNoDict[nID], starting_=nStarting)

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
    with open("poiNetwork/NY_CH_es_12.csv", 'r') as rf:
        spamreader = csv.reader(rf)
        next(spamreader)

        count = 1

        for eachE in spamreader:
            eID1, eID2, eW, eIsShort, eMid = int(eachE[0]), int(eachE[1]), float(eachE[2]), eachE[3], int(eachE[4])

            if math.isnan(eW):
                continue

            if eIsShort == 'N':
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

    ###############################################################
    gc = {}

    pool = mp.Pool(mp.cpu_count())

    ###container = pool.map(g.containerCal, [i for i in g.nodes])
    container = pool.map(g.containerCal2, [i for i in g.nodes])

    pool.close()

    for nID, vRes in container:
        gc[nID] = vRes

    with open('CGContainer_12.pickle', 'wb') as f:
        pickle.dump(gc, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()