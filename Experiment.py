import csv, pickle, time
import ContractPoINetwork, CONSTANTS
import GreedySearch
import random
import math
import heapq
import os
import Node


def random_walk_restart(g, origin, theta, max_dist, max_time, num_origins=1, verbal=True, complexity=True):

    if verbal:
        print("===========================================")
        print("Start RWR searching for category: ", ','.join([str(x) for x in theta]), " within distance ", max_dist,
              " within ", max_time, " sec limit")

    start_time = time.time()

    res_summary, res_origins = [], set()
    origin = tuple(origin)

    while time.time() - start_time <= max_time:
        if verbal:
            print("Another new round from Random Walk ...")

        while True:
            cur_node = random.choice(origin)

            if cur_node not in res_origins:
                break

        cur_res = set()

        if len(g.nodes[cur_node].PoIs) > 0:
            satisfied_flag = 0

            for idx in range(CONSTANTS.CATEGORY_NUM):
                # Desire a category and node has corresponding category
                count = min(g.nodes[cur_node].category[idx], theta[idx])

                if count > 0:
                    category_count = 0
                    for each_poi in g.nodes[cur_node].PoIs:
                        if each_poi.category == idx:
                            cur_res.add(each_poi)
                            category_count += 1

                            if category_count >= count:  break

                # Current node has more PoIs then enough
                if count == theta[idx]:  satisfied_flag += 1

            if satisfied_flag == CONSTANTS.CATEGORY_NUM:
                '''
                stop_time = time.time()

                if verbal:  print("Find Everything at Starting Point !!!")

                if complexity:
                    if verbal:
                        print("Diversity: ", len(cur_res), " by using ", stop_time - start_time, " sec and 0 edge !!!")

                    return cur_node, [cur_node], (stop_time - start_time, sum(theta)), (0, sum(theta))
                else:
                    return cur_node, [cur_node], None, None
                '''

                each_res = (cur_node, [cur_node], 0, cur_res)

                if complexity:
                    each_res += (time.time() - start_time, )

                res_summary.append(each_res)

                res_origins.add(cur_node)

                if len(res_origins) >= num_origins:
                    if complexity:
                        return res_summary, time.time() - start_time
                    else:
                        return res_summary

                continue

        cur_path, cur_dist = [cur_node], 0

        need = [x for x in theta]

        for each_poi in cur_res:
            need[each_poi.category] = max(0, need[each_poi.category] - 1)

        if verbal:
            print("Starving for category: ", ','.join([str(x) for x in need]))

        while time.time() - start_time <= max_time:
            if cur_path[0] in res_origins:
                break

            outgoing_edge = []

            for each_edge in g.neighbor_edges[cur_node]:
                if (each_edge.role == CONSTANTS.SRC) and (not math.isnan(each_edge.weight)) and \
                        (cur_dist + each_edge.weight <= max_dist) and (not each_edge.isShortcut):
                    outgoing_edge.append(each_edge)

            if len(outgoing_edge) == 0:
                break

            next_edge = outgoing_edge[random.randint(0, len(outgoing_edge)-1)]
            next_node = next_edge.node_id2

            if len(g.nodes[next_node].PoIs) > 0:
                for idx in range(CONSTANTS.CATEGORY_NUM):
                    # Current node has desired category
                    count = min(need[idx], g.nodes[next_node].category[idx])

                    if count > 0:
                        for each_poi in g.nodes[next_node].PoIs:
                            if (each_poi.category == idx) and (each_poi not in cur_res):
                                cur_res.add(each_poi)

                                count -= 1
                                need[idx] -= 1

                                if count <= 0:
                                    break

                if all([x <= 0 for x in need]):
                    cur_path.append(next_node)

                    '''
                    if verbal:
                        print("Found solution and terminated earlier !!!")
                        print("->".join(str(x) for x in cur_path))
                        print("===========================================")

                    if complexity:
                        stop_time = time.time()
                        return cur_path[0], cur_path, (stop_time - start_time, sum(theta)), (edge_count, sum(theta))
                    else:
                        return cur_path[0], cur_path, None, None
                    '''

                    each_res = (cur_path[0], cur_path, cur_dist + next_edge.weight, cur_res)

                    if complexity:
                        each_res += (time.time() - start_time, )

                    res_summary.append(each_res)

                    res_origins.add(cur_path[0])

                    if len(res_origins) >= num_origins:
                        if complexity:
                            return res_summary, time.time() - start_time
                        else:
                            return res_summary

                    break

            cur_node = next_node
            cur_path.append(next_node)
            cur_dist += next_edge.weight

    if verbal:
        print("In total ", len(res_summary), " results")

    if complexity:
        return res_summary, time.time() - start_time
    else:
        return res_summary


def dijkstra_theta(g, origin, theta, max_dist, res, complexity):
    if complexity:  edge_count = 0

    dist = [float('inf')] * len(g.nodes)
    dist[origin] = 0

    pq = []
    heapq.heappush(pq, (0, origin))

    path = {}
    path[origin] = None

    visited = set()

    while pq:
        cur_dist, cur_node = heapq.heappop(pq)

        if cur_node in visited:  continue

        if cur_dist > max_dist:
            if complexity:
                return -1, float('inf'), [], edge_count # node, distance used, path
            else:
                return -1, float('inf'), [], None

        if len(g.nodes[cur_node].PoIs) > 0:
            for idx in range(CONSTANTS.CATEGORY_NUM):
                # Desire a category and node has corresponding category
                count = min(g.nodes[cur_node].category[idx], theta[idx])

                if count > 0:
                    for each_poi in g.nodes[cur_node].PoIs:
                        if (each_poi.category == idx) and (each_poi not in res):
                            res_head, res_path = cur_node, []

                            while res_head is not None:
                                res_path.append(res_head)
                                res_head = path[res_head]

                            res_path.reverse()

                            if complexity:
                                return cur_node, cur_dist, res_path, edge_count
                            else:
                                return cur_node, cur_dist, res_path, None

        visited.add(cur_node)

        for each_edge in g.neighbor_edges[cur_node]:
            if complexity:  edge_count += 1

            if each_edge.role == CONSTANTS.DEST or math.isnan(each_edge.weight):
                continue

            if cur_dist + each_edge.weight > max_dist:  continue

            next_node, next_dist = each_edge.node_id2, cur_dist + each_edge.weight

            if next_dist < dist[next_node]:
                path[next_node] = cur_node
                dist[next_node] = next_dist

                heapq.heappush(pq, (next_dist, next_node))

    if complexity:
        return -1, float('inf'), [], edge_count  # node, distance used, path
    else:
        return -1, float('inf'), [], None


def greedy_dijkstra(g, origins, theta, max_dist, num_origins=1, verbal=True, complexity=True):
    if verbal:
        print("===========================================")
        print("Start greedy Dijkstra searching for category: ", ','.join([str(x) for x in theta]),
              " within distance ", max_dist, " and # of origins ", num_origins)

    if complexity:
        edge_count = 0
        start_time = time.time()

    res_summary = []

    for each_o in origins:
        res_o = set()
        theta_o = [x for x in theta]

        if len(g.nodes[each_o].PoIs) > 0:
            for idx in range(CONSTANTS.CATEGORY_NUM):
                # Desire a category and node has corresponding category
                count = min(g.nodes[each_o].category[idx], theta_o[idx])

                if count > 0:
                    for each_poi in g.nodes[each_o].PoIs:
                        if each_poi.category == idx:
                            res_o.add(each_poi)
                            theta_o[idx] -= 1

                            if theta_o[idx] == 0:  break

            if sum(theta_o) == 0:
                each_res = (each_o, [each_o], 0, res_o)

                if complexity:
                    each_res += (time.time() - start_time, )

                res_summary.append(each_res)

                if len(res_summary) >= num_origins:
                    if complexity:
                        return res_summary, time.time() - start_time
                    else:
                        return res_summary

                continue

        cur_node = each_o
        cur_path = []
        rest_dist = max_dist

        while True:
            next_nearest_poi_node, next_nearest_dist, sub_seq, edge_complex = \
                dijkstra_theta(g, cur_node, theta_o, rest_dist, res_o, complexity)

            if complexity:  edge_count += edge_complex

            if next_nearest_dist == float('inf'):  break

            if len(cur_path) == 0:
                cur_path = sub_seq
            else:
                cur_path = cur_path[:-1] + sub_seq

            for idx in range(CONSTANTS.CATEGORY_NUM):
                # Desire a category and node has corresponding category
                count = min(g.nodes[next_nearest_poi_node].category[idx], theta_o[idx])

                if count > 0:
                    for each_poi in g.nodes[next_nearest_poi_node].PoIs:
                        if (each_poi.category == idx) and (each_poi not in res_o):
                            res_o.add(each_poi)
                            theta_o[idx] -= 1

                            if theta_o[idx] == 0:  break

            if sum(theta_o) == 0:
                '''
                if verbal:
                    print("Found solution and terminated earlier !!!")
                    print("->".join(str(x) for x in cur_path))
                    print("===========================================")

                if complexity:
                    stop_time = time.time()
                    return each_o, cur_path, (stop_time - start_time, sum(theta)), (edge_count, sum(theta))
                else:
                    return each_o, cur_path, None, None
                '''

                if verbal:
                    print("Found an solution !!! In total, ", len(res_summary) + 1, " found")
                    print("->".join(str(x) for x in cur_path))
                    print("===========================================")

                each_res = (each_o, cur_path, max_dist - rest_dist + next_nearest_dist, res_o)

                if complexity:
                    each_res += (time.time() - start_time, )

                res_summary.append(each_res)

                if len(res_summary) >= num_origins:
                    if complexity:
                        return res_summary, time.time() - start_time
                    else:
                        return res_summary

                break

            rest_dist -= next_nearest_dist
            cur_node = next_nearest_poi_node

    if verbal:
        print("In total ", len(res_summary), " results")

    if complexity:
        return res_summary, time.time() - start_time
    else:
        return res_summary


def calculate_path_length(g, path):
    res = 0

    for idx in range(len(path)-1):
        node_1, node_2 = path[idx], path[idx+1]
        res += g.edges[(node_1, node_2)].weight

    return res


def main():
    num_poi = 120

    if os.path.exists("PoI_Network/PKL/NY_" + str(num_poi) + "_PoI_network_euclidean.pickle"):
        print("Start Loading PoI Network from Pickle file......")

        with open('PoI_Network/PKL/NY_PoI_network_euclidean.pickle', 'rb') as f:
            g = pickle.load(f)

        print("PoI Network Has Been Loaded......")
        print("=================================================")
    elif os.path.exists("PoI_Network/CSV/NY_CH_es_euclidean.csv") and \
            os.path.exists("PoI_Network/CSV/NY_CH_ns_euclidean.csv") and \
            os.path.exists("PoI_Network/CSV/NY_PoI_info.csv"):
        print("Start Loading PoI Network from CSV files......")

        print("Start Loading Diversity......")

        with open("PoI_Network/CSV/NY_PoI_info.csv", 'r') as rf:
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
                        node_pois.add(poi_dict[each_poi_name])

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
                    g.add_node(node_id, node_lng, node_lng)
                else:
                    g.add_node(node_id, node_lng, node_lng, embedded_poi_node[node_id])

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

                if math.isnan(edge_weight):
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
    else:
        print("No PoI network data!!!")
        exit()

    ############################################################################################
    ############################################################################################
    ############################################################################################

    if os.path.exists("PoI_Network/Index/MatrixContainer_" + str(num_poi) + ".pickle"):
        print("Start Loading Container Index from Pickle file......")

        with open("PoI_Network/Index/MatrixContainer_" + str(num_poi) + ".pickle", 'rb') as f:
            container_index = pickle.load(f)

        print("Container Index Has Been Loaded......")
        print("=================================================")
    else:
        print("No Index Found!!!")
        exit()

    ############################################################################################
    ############################################################################################
    ############################################################################################
    if os.path.exists("PoI_Network/NY_ns.csv"):
        print("Starting Locating Origins/Hotels......")
        origins = set()

        with open("PoI_Network/NY_ns.csv", 'r') as rf:
            spamreader = csv.reader(rf)
            next(spamreader)

            for each_row in spamreader:
                node_id, hotel_flag = int(each_row[0]), each_row[4]

                if hotel_flag == 'Y':  origins.add(node_id)

        print("In total ", len(origins), " Origins/Hotels found......")
        print("==================================================")
    else:
        print("No Hotel/Origin Info!!!")
        exit()

    # Generate new sample from all origins
    new_sample_origin = False
    # If want sampled subset of origins OR all origins
    sampled_origin = False

    baseline_greedy_dijkstra = True
    baseline_rwr = True

    if sampled_origin:
        num_origin = 120
    else:
        num_origin = len(origins)

    if new_sample_origin:
        origins = random.sample(list(origins), k=num_origin)

        with open('ExperimentRelated/random' + str(num_origin) + '.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow([str(x) for x in origins])

        origins = set(origins)

        print("Origins Sampling Completed...")
        print("==================================================")
    elif sampled_origin:
        with open('ExperimentRelated/random' + str(num_origin) + '.csv', 'r') as f:
            rf = csv.reader(f)

            for row in rf:
                if not row:  break
                origins = set([int(x) for x in row])

        print("Sampled Origins Loaded Completed...")
        print("==================================================")
    else:
        print("Use All ", len(origins), " Origins")
        print("==================================================")

    g.rtree_build(origins)

    theta_list = []

    with open('ExperimentRelated/randomVar.csv', 'r') as f:
        reader = csv.reader(f)

        for each_theta in reader:
            theta_list.append([int(x) for x in each_theta])

    #time_result_PoI, edge_result_PoI, time_result_origin, edge_result_origin = {}, {}, {}, {}

    #time_result_greedy_dijkstra, edge_result_greedy_dijkstra = {}, {}

    if os.path.exists("ExperimentResult/timeResultVariantPoI_" + str(num_origin) + ".pickle"):
        with open("ExperimentResult/timeResultVariantPoI_" + str(num_origin) + ".pickle", 'rb') as f:
            time_result_PoI = pickle.load(f)
    else:
        time_result_PoI = {}

    if os.path.exists("ExperimentResult/timeResultVariantOrigin_" + str(num_origin) + ".pickle"):
        with open("ExperimentResult/timeResultVariantOrigin_" + str(num_origin) + ".pickle", 'rb') as f:
            time_result_origin = pickle.load(f)
    else:
        time_result_origin = {}

    if os.path.exists("ExperimentResult/timeResultGreedyDijkstra_" + str(num_origin) + ".pickle"):
        with open("ExperimentResult/timeResultGreedyDijkstra_" + str(num_origin) + ".pickle", 'rb') as f:
            time_result_greedy_dijkstra = pickle.load(f)
    else:
        time_result_greedy_dijkstra = {}

    if os.path.exists("ExperimentResult/timeTripleResultRandomWalk_" + str(num_origin) + ".pickle"):
        with open("ExperimentResult/timeTripleResultRandomWalk_" + str(num_origin) + ".pickle", 'rb') as f:
            time_result_rwr_triple = pickle.load(f)
    else:
        time_result_rwr_triple = {}

    required_origins = [1, 5, 20, 50]

    dist_limit = [500, 1000, 2000, 4000, 8000]

    count = 0

    if not os.path.exists("ExperimentResult"):
        os.mkdir("ExperimentResult")

    for num_required_origin in required_origins:
        for theta in theta_list:
            for max_dist in dist_limit:
                print("==================================================")
                print("Now processing ", theta, " with distance ", max_dist, " and # of origins ", num_required_origin)
                print(count + 1, "/", len(required_origins)*len(theta_list)*len(dist_limit))

                if baseline_greedy_dijkstra:
                    if (tuple(theta) + (max_dist,) + (num_required_origin,)) not in time_result_greedy_dijkstra:
                        print("By using Greedy Dijkstra......")

                        gd_res, gd_time_cost = \
                            greedy_dijkstra(g, origins, theta, max_dist, num_required_origin,
                                            verbal=False, complexity=True)

                        if len(gd_res) == 0:
                            print("Found NOTHING by using ", gd_time_cost, "s")
                        elif len(gd_res) < num_required_origin:
                            print("FAIL! ONLY FOUND ", len(gd_res), " origins by using ", gd_time_cost, "s")
                        else:
                            print("SUCCESS! FOUND ALL ", len(gd_res), " origins by using ", gd_time_cost, "s")

                        time_result_greedy_dijkstra[tuple(theta) + (max_dist,) + (num_required_origin,)] = \
                            (gd_res, gd_time_cost)

                        t_result_name = 'ExperimentResult/timeResultGreedyDijkstra' + '_' + str(num_origin)

                        with open(t_result_name + '.pickle', 'wb') as f:
                            pickle.dump(time_result_greedy_dijkstra, f, pickle.HIGHEST_PROTOCOL)

                if (tuple(theta) + (max_dist,) + (num_required_origin,)) not in time_result_PoI:
                    print("Starting from PoI......")

                    alg_poi_res, alg_poi_time_cost = \
                        GreedySearch.greedy_process_PoI(g, container_index, theta, max_dist, origins, num_required_origin,
                                                        index_matrix=True, verbal=False, complexity=True)

                    if len(alg_poi_res) == 0:
                        print("Found NOTHING by using ", alg_poi_time_cost, "s")
                    elif len(alg_poi_res) < num_required_origin:
                        print("FAIL! ONLY FOUND ", len(alg_poi_res), " origins by using ", alg_poi_time_cost, "s")
                    else:
                        print("SUCCESS! FOUND ALL ", len(alg_poi_res), " origins by using ", alg_poi_time_cost, "s")

                    time_result_PoI[tuple(theta) + (max_dist,) + (num_required_origin,)] = (alg_poi_res, alg_poi_time_cost)

                    t_poi_result_name = 'ExperimentResult/timeResultVariantPoI' + '_' + str(num_origin)

                    with open(t_poi_result_name + '.pickle', 'wb') as f:
                        pickle.dump(time_result_PoI, f, pickle.HIGHEST_PROTOCOL)

                if (tuple(theta) + (max_dist,) + (num_required_origin,)) not in time_result_origin:
                    print("Starting from Origin......")

                    alg_o_res, alg_o_time_cost = \
                        GreedySearch.greedy_process_origin(g, container_index, theta, max_dist, origins, num_required_origin,
                                                           index_matrix=True, verbal=False, complexity=True)

                    if len(alg_o_res) == 0:
                        print("Found NOTHING by using ", alg_o_time_cost, "s")
                    elif len(alg_o_res) < num_required_origin:
                        print("FAIL! ONLY FOUND ", len(alg_o_res), " origins by using ", alg_o_time_cost, "s")
                    else:
                        print("SUCCESS! FOUND ALL ", len(alg_o_res), " origins by using ", alg_o_time_cost, "s")

                    time_result_origin[tuple(theta) + (max_dist,) + (num_required_origin,)] = (alg_o_res, alg_o_time_cost)

                    t_origin_result_name = 'ExperimentResult/timeResultVariantOrigin' + '_' + str(num_origin)

                    with open(t_origin_result_name + '.pickle', 'wb') as f:
                        pickle.dump(time_result_origin, f, pickle.HIGHEST_PROTOCOL)

                count += 1

                if baseline_rwr:
                    if not baseline_greedy_dijkstra:
                        t_gd_result_name = 'ExperimentResult/timeResultGreedyDijkstra' + '_' + str(num_origin)

                        with open(t_gd_result_name + ".pickle", 'rb') as f:
                            time_result_greedy_dijkstra = pickle.load(f)

                    max_time = max([time_result_greedy_dijkstra[tuple(theta) + (max_dist,) + (num_required_origin,)][1],
                                    time_result_PoI[tuple(theta) + (max_dist,) + (num_required_origin,)][1],
                                    time_result_origin[tuple(theta) + (max_dist,) + (num_required_origin,)][1]
                                    ])

                    '''
                    print("Random Walk with Restart with time limit ", max_time)

                    rwr_res, rwr_time_cost = \
                        random_walk_restart(g, origins, theta, max_dist, max_time, num_required_origin,
                                            verbal=False, complexity=True)

                    if len(rwr_res) == 0:
                        print("Found NOTHING by using ", rwr_time_cost, "s")
                    elif len(rwr_res) < num_required_origin:
                        print("FAIL! ONLY FOUND ", len(rwr_res), " origins by using ", rwr_time_cost, "s")
                    else:
                        print("SUCCESS! FOUND ALL ", len(rwr_res), " origins by using ", rwr_time_cost, "s")

                    time_result_rwr[tuple(theta) + (max_dist,) + (num_required_origin,)] = (rwr_res, rwr_time_cost)
                    '''
                    if (tuple(theta) + (max_dist,) + (num_required_origin,)) not in time_result_rwr_triple:
                        print("Random Walk with Restart with triple time limit ", 3*max_time)

                        rwr_res, rwr_time_cost = \
                            random_walk_restart(g, origins, theta, max_dist, 3*max_time, num_required_origin,
                                                verbal=False, complexity=True)

                        if len(rwr_res) == 0:
                            print("Found NOTHING by using ", rwr_time_cost, "s")
                        elif len(rwr_res) < num_required_origin:
                            print("FAIL! ONLY FOUND ", len(rwr_res), " origins by using ", rwr_time_cost, "s")
                        else:
                            print("SUCCESS! FOUND ALL ", len(rwr_res), " origins by using ", rwr_time_cost, "s")

                        time_result_rwr_triple[tuple(theta) + (max_dist,) + (num_required_origin,)] = \
                            (rwr_res, rwr_time_cost)

                        print("Done ", count, "/", len(time_result_PoI))
                        print("==================================================")

                        #t_result_name = 'ExperimentResult/timeResultRandomWalk' + '_' + str(num_origin)
                        t_triple_result_name = 'ExperimentResult/timeTripleResultRandomWalk' + '_' + str(num_origin)

                        #with open(t_result_name + '.pickle', 'wb') as f:
                        #    pickle.dump(time_result_rwr, f, pickle.HIGHEST_PROTOCOL)

                        with open(t_triple_result_name + '.pickle', 'wb') as f:
                            pickle.dump(time_result_rwr_triple, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()