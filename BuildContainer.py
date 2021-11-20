import pickle
import multiprocessing as mp
import csv, math, os
import time

import ContractPoINetwork, CONSTANTS
import Node


def main():
    num_poi = 623

    if os.path.exists("PoI_Network/PKL/NY_" + str(num_poi) + "_PoI_network_euclidean.pickle"):
        print("Start Loading PoI Network from Pickle file......")

        with open("PoI_Network/PKL/NY_" + str(num_poi) + "_PoI_network_euclidean.pickle", 'rb') as f:
            g = pickle.load(f)

        print("PoI Network Has Been Loaded......")
        print("=================================================")
    elif os.path.exists("PoI_Network/CSV/NY_CH_es_euclidean.csv") and \
            os.path.exists("PoI_Network/CSV/NY_CH_ns_euclidean.csv") and \
            os.path.exists("PoI_Network/CSV/NY_" + str(num_poi) + "_PoI_info.csv"):
        print("Start Loading PoI Network from CSV files......")

        print("Start Loading Diversity......")

        with open("PoI_Network/CSV/NY_" + str(num_poi) + "_PoI_info.csv", 'r') as rf:
            spamreader = csv.reader(rf)
            next(spamreader)

            poi_dict = {}

            for eachPoI in spamreader:
                poi_id, poi_name, category_idx = int(eachPoI[0]), eachPoI[1], int(eachPoI[2])

                poi_dict[poi_name] = Node.PoI(poi_id=poi_id, name=poi_name, category=category_idx)

        ############################################################################################
        ############################################################################################
        ############################################################################################

        print("Start Pairing Node with PoIs......")

        embedded_poi_node = {}

        '''
        {
            node_1: set(Node.PoI(), Node.PoI(), ...),
            ...
        }
        Note: Only contain the node with PoI embedded
        This step is necessary because CH_ns does not have info related to PoI
        '''

        with open("PoI_Network/NY_ns.csv", 'r') as rf:
            spamreader = csv.reader(rf)
            next(spamreader)

            for each_node in spamreader:
                node_id, node_lng, node_lat, node_pois_str = int(each_node[0]), float(each_node[1]), \
                                                             float(each_node[2]), each_node[3]

                if node_pois_str != '':
                    node_pois = set()
                    pois_name_list = node_pois_str.split('|')

                    for each_poi_name in pois_name_list:
                        if each_poi_name not in poi_dict:  continue

                        node_pois.add(poi_dict[each_poi_name])

                    if node_pois:
                        embedded_poi_node[node_id] = node_pois

        ############################################################################################
        ############################################################################################
        ############################################################################################

        g = ContractPoINetwork.ContractPoINetwork()

        print("Start Inserting Nodes......")
        with open("PoI_Network/CSV/NY_CH_ns_euclidean.csv", 'r') as rf:
            spamreader = csv.reader(rf)
            next(spamreader)

            counter = 1

            for each_row in spamreader:
                node_id, node_lng, node_lat, node_depth, node_order = int(each_row[0]), float(each_row[1]), \
                                                                      float(each_row[2]), int(each_row[3]), \
                                                                      int(each_row[4])

                if node_id not in embedded_poi_node:
                    g.add_node(node_id, lng_=node_lng, lat_=node_lat)
                else:
                    g.add_node(node_id, lng_=node_lng, lat_=node_lat, pois_=embedded_poi_node[node_id])

                '''
                :param depth: Integer, Hierarchy Depth
                :param contractOrder: Integer, contract order
                '''

                g.nodes[node_id].depth, g.nodes[node_id].contract_order = node_depth, node_order

                print("Inserted ", counter, " nodes")
                g.nodes[node_id].print_info()
                print("///////////////////////")
                counter += 1

        print("Inserted all ", counter - 1, " nodes successfully......")

        ############################################################################################
        ############################################################################################
        ############################################################################################

        print("Starting Inserting Edges......")
        with open("PoI_Network/CSV/NY_CH_es_euclidean.csv", 'r') as rf:
            spamreader = csv.reader(rf)
            next(spamreader)

            counter = 1

            for each_row in spamreader:
                node_id1, node_id2, edge_weight, edge_isShortcut, mid_node_id = int(each_row[0]), int(each_row[1]), \
                                                                                float(each_row[2]), each_row[3], \
                                                                                int(each_row[4])

                if math.isnan(edge_weight) or edge_isShortcut != 'N':
                    continue

                if edge_isShortcut == 'N':
                    g.add_edge(node_id1, node_id2, edge_weight)
                    print("Inserted ", counter, " edges")
                else:
                    g.add_shortcut(node_id1, node_id2, edge_weight, mid_node_id)
                    print("Inserted ", counter, " shortcuts")

                g.edges[(node_id1, node_id2)].print_info()
                print("///////////////////////")
                counter += 1

        print("Inserted all ", counter - 1, " edges successfully......")
        print("==================================================")

        with open("PoI_Network/PKL/NY_" + str(num_poi) + "_PoI_network_euclidean.pickle", 'wb') as f:
            pickle.dump(g, f, pickle.HIGHEST_PROTOCOL)
        exit()
    else:
        print("No PoI network data!!!")
        exit()

    ############################################################################################
    ############################################################################################
    ############################################################################################

    gc = {}

    time_record = []

    start_t = time.time()

    pool = mp.Pool(mp.cpu_count())

    ###container = pool.map(g.containerCal, [i for i in g.nodes])
    container = pool.map(g.matrix_container_cal, [i for i in g.nodes])

    pool.close()

    for node_id, gc_res, n_t in container:
        gc[node_id] = gc_res
        time_record.append(n_t)

    if not os.path.exists("PoI_Network/Index"):
        os.mkdir("PoI_Network/Index")

    print("avg. time", sum(time_record)/len(time_record))

    print("in total time", time.time() - start_t)
    #with open("PoI_Network/Index/MatrixContainer_" + str(num_poi) + ".pickle", 'wb') as f:
    #    pickle.dump(gc, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()