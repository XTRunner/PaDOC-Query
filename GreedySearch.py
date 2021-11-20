import time
import heapq
from functools import total_ordering
import CONSTANTS
import math
import collections
from geopy.distance import geodesic


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
        # Priority: the less, the better
        # Distance ( to origin): the less, the better
        return (self.priority, self.dist) < \
               (other.priority, other.dist)

    def print_path(self):
        path, cur = [], self

        while cur:
            path.append(cur.nid)

            cur = cur.parent

        path.reverse()

        return path

    def find_head(self):
        head, cur = None, self

        while cur:
            head = cur.nid
            cur = cur.parent

        return head


def meter_to_degree(dist, lat, lng):
    rest_cost = dist / CONSTANTS.REarth

    d_lat = rest_cost
    d_lng = rest_cost / math.cos(CONSTANTS.varPiDegree * lat)

    d_lat = d_lat / CONSTANTS.varPiDegree
    d_lng = d_lng / CONSTANTS.varPiDegree

    lng_min = min(lng + d_lng, lng - d_lng)
    lng_max = max(lng + d_lng, lng - d_lng)
    lat_min = min(lat + d_lat, lat - d_lat)
    lat_max = max(lat + d_lat, lat - d_lat)

    return lng_min, lng_max, lat_min, lat_max


def greedy_process_PoI(g, container_index, theta, max_dist, origin, num_origins=1, index_matrix=False,
                       verbal=True, complexity=True):
    """
    :param g: PoI network
    :param container_index: geometric container
    :param theta: categorical require [1, 2, 0, 2, 3, 1]
    :param max_dist: distance limit
    :param origin: set(), Node ID of starting query locations
    :param num_origins: preferred number of origins
    :param index_matrix: index is vector or matrix
    :param verbal: Print out information while running
    :param complexity: Return time and edge complexity
    """
    if verbal:
        print("===========================================")
        print("......Starting PoI Version......")
        print("Start searching for category: ", ','.join([str(x) for x in theta]), " within distance ", max_dist)
        print("Number of starting points: ", len(origin))

    if complexity:
        start_time = time.time()

    res_summary, res_origins = [], set()

    pq = []

    # Store the largest query range that had been tried
    # Any smaller range can be ignored immediately
    queried = collections.defaultdict(lambda : -1)
    euclidean_approx = {}

    for node_id, val in g.nodes.items():
        # Node  --  PoIs: set([PoI(id1, name1, category1), PoI(id2, name2, category2),... ])
        #           category: [0, 0, 1, 2, 0, 0]
        if len(val.PoIs) > 0:
            stock = set()
            satisfied_flag = 0

            for idx in range(CONSTANTS.CATEGORY_NUM):
                # Desire a category and node has corresponding category
                count = min(val.category[idx], theta[idx])

                if count > 0:
                    category_count = 0

                    for each_poi in val.PoIs:
                        if each_poi.category == idx:
                            stock.add(each_poi)
                            category_count += 1

                            if category_count >= count:
                                break

                # Current node has more PoIs then enough
                if count == theta[idx]:
                    satisfied_flag += 1

            if len(stock) == 0:
                continue

            # Ignore those PoIs with 0 origin in Euclidean space
            lng_min, lng_max, lat_min, lat_max = meter_to_degree(max_dist, lat=val.lat, lng=val.lng)

            range_res = list(g.tree.intersection((lng_min, lat_min, lng_max, lat_max), objects="raw"))

            if len(range_res) == 0:
                continue

            if satisfied_flag == CONSTANTS.CATEGORY_NUM:
                if verbal:
                    print("Find Everything at a PoI location !!!")

                queried[node_id] = max_dist

                candidates = set(range_res)

                candidates = candidates.difference(res_origins)

                if verbal:
                    print("Range query of ", val.lng, val.lat, " with budget ", max_dist)
                    print("i.e., bounding box -- ", lng_min, lng_max, lat_min, lat_max)
                    print("Found in total ", len(candidates), " candidates...")

                origin_to_PoI_res = \
                    g.multi_origins_single_target_reverse_dijkstra(candidates, node_id, max_dist,
                                                                   num_origins-len(res_origins), res_origins)

                if len(origin_to_PoI_res) != 0:
                    if verbal:
                        print("Found ", len(origin_to_PoI_res), " new origins")
                        print("In total ", len(origin_to_PoI_res) + len(res_summary), " results")

                    for (each_origin_to_PoI, each_dist) in origin_to_PoI_res:
                        if each_origin_to_PoI[0] in res_origins:
                            continue

                        each_res = (each_origin_to_PoI[0], each_origin_to_PoI, each_dist, stock)

                        if complexity:
                            each_res += (time.time() - start_time, )

                        res_summary.append(each_res)

                        res_origins.add(each_origin_to_PoI[0])

                    if len(res_origins) >= num_origins:
                        if complexity:
                            return res_summary, time.time() - start_time
                        else:
                            return res_summary
            else:
                nearest_origin = list(g.tree.nearest((val.lng, val.lat, val.lng, val.lat), objects='raw'))

                euclidean_dist = geodesic((val.lat, val.lng),
                                          (g.nodes[nearest_origin[0]].lat, g.nodes[nearest_origin[0]].lng)).m

                if euclidean_dist > max_dist:
                    continue

                euclidean_approx[node_id] = euclidean_dist

                if verbal:
                    print("Find a starting point ...... In total ", len(pq), " starting points now")

                heapq.heappush(pq, Element(nid=node_id, res=stock, priority=euclidean_dist, dist=0))

    while pq:
        if verbal:
            print(len(pq), " elements awaiting in queue...")

        cur_element = heapq.heappop(pq)

        src_head = cur_element.find_head()

        if max_dist - cur_element.dist <= queried[src_head]:
            continue

        need = [x for x in theta]

        for each_poi in cur_element.res:
            need[each_poi.category] -= 1

        if (cur_element.priority <= max_dist) and (cur_element.parent is not None):
            rest_cost = max_dist - (cur_element.priority - euclidean_approx[src_head])

            lng_min, lng_max, lat_min, lat_max = meter_to_degree(rest_cost,
                                                                 lat=g.nodes[src_head].lat,
                                                                 lng=g.nodes[src_head].lng)

            if any([math.isnan(lng_min), math.isnan(lng_max), math.isnan(lat_min), math.isnan(lat_max)]):
                continue

            range_res = list(g.tree.intersection((lng_min, lat_min, lng_max, lat_max), objects="raw"))

            candidates = set(range_res)

            if verbal:
                print("Range query of ", g.nodes[src_head].lng, g.nodes[src_head].lat,
                      " with budget ", rest_cost)
                print("i.e., bounding box -- ", lng_min, lng_max, lat_min, lat_max)
                print("Found in total ", len(candidates), " candidates...")

            candidates = candidates.difference(res_origins)

            if len(candidates) > 0:
                cur_res_collected = set([x for x in cur_element.res])
                cur_res_need = [x for x in need]

                rest_path, rest_dist, total_res = retrieve_greedy_dijkstra(g, cur_element.nid, cur_res_need,
                                                                           max_dist - (cur_element.dist
                                                                                       + euclidean_approx[src_head]),
                                                                           cur_res_collected,
                                                                           verbal=False)

                if (rest_path is not None) \
                        and (euclidean_approx[src_head] + cur_element.dist + rest_dist <= max_dist)  \
                        and (max_dist - (cur_element.dist + rest_dist) > queried[src_head]):
                    rest_cost = max_dist - (cur_element.dist + rest_dist)

                    queried[src_head] = rest_cost

                    origin_to_PoI_res = g.multi_origins_single_target_reverse_dijkstra(candidates, src_head, rest_cost,
                                                                                       num_origins - len(res_origins),
                                                                                       res_origins)

                    if len(origin_to_PoI_res) != 0:
                        if verbal:
                            print("Found ", len(origin_to_PoI_res), " new origins")
                            print("In total ", len(origin_to_PoI_res) + len(res_summary), " results")

                        for (each_origin_to_PoI, each_dist) in origin_to_PoI_res:
                            if each_origin_to_PoI[0] in res_origins:
                                continue

                            each_res = (each_origin_to_PoI[0],
                                        each_origin_to_PoI[:-1] + cur_element.print_path() + rest_path[1:],
                                        each_dist + cur_element.dist + rest_dist,
                                        total_res)

                            if complexity:
                                each_res += (time.time() - start_time,)

                            res_summary.append(each_res)

                            res_origins.add(each_origin_to_PoI[0])

                        if len(res_origins) >= num_origins:
                            if complexity:
                                return res_summary, time.time() - start_time
                            else:
                                return res_summary

        if verbal:
            print("Starving for category: ", ','.join([str(x) for x in need]))

        for each_edge in g.neighbor_edges[cur_element.nid]:
            if each_edge.role == CONSTANTS.DEST or math.isnan(each_edge.weight) or each_edge.isShortcut:
                continue

            adj_node = each_edge.node_id2

            cur_element_check = cur_element
            skip_no_PoI, loop_found = True, False

            while cur_element_check.parent:
                if g.nodes[cur_element_check.nid].PoIs:
                    skip_no_PoI = False
                    break

                if adj_node == cur_element_check.parent.nid:
                    loop_found = True
                    break

                cur_element_check = cur_element_check.parent

            if skip_no_PoI and loop_found:  continue

            if (cur_element.nid, adj_node) not in cur_element.explored:
                if not index_matrix:
                    next_min_dist = 0
                    next_max_dist = cur_element.dist + each_edge.weight

                    for idx in range(CONSTANTS.CATEGORY_NUM):
                        if need[idx] > 0:
                            next_min_dist = max(next_min_dist, container_index[adj_node][idx])
                            next_max_dist += 2 * container_index[adj_node][idx]

                    if cur_element.dist + each_edge.weight + next_min_dist > max_dist:
                        continue
                else:
                    cur_collection = set([x for x in cur_element.res])

                    next_min_dist = 0

                    for idx in range(CONSTANTS.CATEGORY_NUM):
                        if next_min_dist == float('inf'):
                            break

                        if need[idx] > 0:
                            num_required_poi = need[idx]
                            satisfied_idx = 0

                            while num_required_poi > 0:
                                if container_index[adj_node][idx][satisfied_idx][0] is None:
                                    next_min_dist = float('inf')
                                    break
                                if container_index[adj_node][idx][satisfied_idx][2].poi_id not in \
                                        set([x.poi_id for x in cur_collection]):
                                    next_min_dist = max(next_min_dist, container_index[adj_node][idx][satisfied_idx][1])
                                    num_required_poi -= 1

                                satisfied_idx += 1

                    if euclidean_approx[src_head] + cur_element.dist + each_edge.weight + next_min_dist > max_dist:
                        continue

                    next_max_dist = euclidean_approx[src_head] + cur_element.dist + each_edge.weight
                    cur_node_id = adj_node
                    num_required_poi = [x for x in need]

                    while any(num_required_poi):
                        next_nearest_dist, picked_node = float('inf'), None

                        for idx in range(CONSTANTS.CATEGORY_NUM):
                            if num_required_poi[idx] > 0:
                                for each_poi_record in container_index[cur_node_id][idx]:
                                    if each_poi_record[0] is None:
                                        break

                                    if (each_poi_record[2].poi_id not in set([x.poi_id for x in cur_collection])) and \
                                            (each_poi_record[1] < next_nearest_dist):
                                        next_nearest_dist, picked_node = each_poi_record[1], each_poi_record
                                        break

                        if picked_node is None:
                            next_max_dist = float('inf')
                            break

                        next_max_dist += next_nearest_dist
                        cur_collection.add(picked_node[2])
                        cur_node_id = picked_node[0]
                        num_required_poi[picked_node[2].category] -= 1

                #print(src_head, cur_element.dist,
                #      euclidean_approx[src_head] + cur_element.dist + each_edge.weight + next_min_dist,
                #      next_max_dist,
                #      len(cur_element.res), cur_element.print_path())
                #print(len(res_origins))

                adj_res = set([x for x in cur_element.res])

                # Add necessary PoIs into .res
                # If current node does not have any PoI, then ignore
                if len(g.nodes[adj_node].PoIs) > 0:
                    for idx in range(CONSTANTS.CATEGORY_NUM):
                        # Current node has desired category
                        count = min(need[idx], g.nodes[adj_node].category[idx])

                        if count > 0:
                            for each_poi in g.nodes[adj_node].PoIs:
                                if (each_poi.category == idx) and \
                                        (each_poi.poi_id not in set([x.poi_id for x in adj_res])):
                                    adj_res.add(each_poi)

                                    count -= 1

                                    if count <= 0:
                                        break

                adj_dist = cur_element.dist + each_edge.weight

                rest_cost = max_dist - adj_dist

                if queried[src_head] >= rest_cost:
                    continue

                adj_explored = set([x for x in cur_element.explored])
                adj_explored.add((cur_element.nid, adj_node))

                if not index_matrix:
                    # min: adj_dist + next_min_dist
                    # max: next_max_dist - next_min_dist
                    # 0.8 * min + 0.2 * max =
                    #   0.8 * adjDist + 0.8 * nextMinDist + 0.2 * nextMaxDist - 0.2 * NextMinDist
                    factor_priority = CONSTANTS.factorPriorVector ** sum(need)
                    adj_element = Element(nid=adj_node, priority=factor_priority * adj_dist +
                                                                 (1 - factor_priority) * next_max_dist +
                                                                 (2 * factor_priority - 1) * next_min_dist,
                                          dist=adj_dist, res=adj_res, parent=cur_element, explored=adj_explored)
                else:
                    # min: adj_dist + next_min_dist
                    # max: next_max_dist
                    # 0.5 * min + 0.5 * max
                    # factor_priority * (adj_dist + next_min_dist) + (1 - factor_priority) * next_max_dist,
                    #if next_max_dist == float('inf'):  adj_priority = next_max_dist
                    #else:
                    #    factor_priority = CONSTANTS.factorPriorMatrix ** sum(need)
                    #    adj_priority = factor_priority * (adj_dist + next_min_dist) + \
                    #                   (1 - factor_priority) * next_max_dist

                    adj_element = Element(nid=adj_node, priority=next_max_dist,
                                          dist=adj_dist, res=adj_res, parent=cur_element, explored=adj_explored)

                # Check the diversity to see if we found the results
                adj_div = [0] * CONSTANTS.CATEGORY_NUM

                for each_poi in adj_res:
                    adj_div[each_poi.category] += 1

                terminate_flag = True

                for idx in range(CONSTANTS.CATEGORY_NUM):
                    if adj_div[idx] < theta[idx]:
                        # If our collections is less than the requirement for category, then searching continues
                        terminate_flag = False
                        break

                if terminate_flag:
                    # rtree
                    queried[src_head] = rest_cost

                    lng_min, lng_max, lat_min, lat_max = meter_to_degree(rest_cost,
                                                                         lat=g.nodes[src_head].lat,
                                                                         lng=g.nodes[src_head].lng)

                    if any([math.isnan(lng_min), math.isnan(lng_max), math.isnan(lat_min), math.isnan(lat_max)]):
                        continue

                    range_res = list(g.tree.intersection((lng_min, lat_min, lng_max, lat_max), objects="raw"))

                    candidates = set(range_res)

                    if verbal:
                        print("Range query of ", g.nodes[src_head].lng, g.nodes[src_head].lat,
                              " with budget ", max_dist - adj_dist)
                        print("i.e., bounding box -- ", lng_min, lng_max, lat_min, lat_max)
                        print("Found in total ", len(candidates), " candidates...")

                    candidates = candidates.difference(res_origins)

                    if len(candidates) > 0:
                        origin_to_PoI_res = g.multi_origins_single_target_reverse_dijkstra(candidates, src_head,
                                                                                           max_dist - adj_dist,
                                                                                           num_origins-len(res_origins),
                                                                                           res_origins)

                        if len(origin_to_PoI_res) != 0:
                            if verbal:
                                print("Found ", len(origin_to_PoI_res), " new origins")
                                print("In total ", len(origin_to_PoI_res) + len(res_summary), " results")

                            for (each_origin_to_PoI, each_dist) in origin_to_PoI_res:
                                if each_origin_to_PoI[0] in res_origins:
                                    continue

                                each_res = (each_origin_to_PoI[0], each_origin_to_PoI[:-1] + adj_element.print_path(),
                                            each_dist + adj_dist, adj_res)

                                if complexity:
                                    each_res += (time.time() - start_time, )

                                res_summary.append(each_res)

                                res_origins.add(each_origin_to_PoI[0])

                            if len(res_origins) >= num_origins:
                                if complexity:
                                    return res_summary, time.time() - start_time
                                else:
                                    return res_summary
                else:
                    heapq.heappush(pq, adj_element)

    if verbal:
        print("In total ", len(res_summary), " results")

    if complexity:
        return res_summary, time.time() - start_time
    else:
        return res_summary


def dijkstra_theta(g, origin, theta, max_dist, res):
    dist = [float('inf')] * len(g.nodes)
    dist[origin] = 0

    pq = []
    heapq.heappush(pq, (0, origin))

    path = {origin: None}

    visited = set()

    cur_res = set([x.poi_id for x in res])

    while pq:
        cur_dist, cur_node = heapq.heappop(pq)

        if cur_node in visited:
            continue

        if cur_dist > max_dist:
            return -1, float('inf'), []

        if len(g.nodes[cur_node].PoIs) > 0:
            for idx in range(CONSTANTS.CATEGORY_NUM):
                # Desire a category and node has corresponding category
                count = min(g.nodes[cur_node].category[idx], theta[idx])

                if count > 0:
                    for each_poi in g.nodes[cur_node].PoIs:
                        if (each_poi.category == idx) and (each_poi.poi_id not in cur_res):
                            res_head, res_path = cur_node, []

                            while res_head is not None:
                                res_path.append(res_head)
                                res_head = path[res_head]

                            res_path.reverse()

                            return cur_node, cur_dist, res_path

        visited.add(cur_node)

        for each_edge in g.neighbor_edges[cur_node]:
            if each_edge.role == CONSTANTS.DEST or math.isnan(each_edge.weight):
                continue

            if cur_dist + each_edge.weight > max_dist:
                continue

            next_node, next_dist = each_edge.node_id2, cur_dist + each_edge.weight

            if next_dist < dist[next_node]:
                path[next_node] = cur_node
                dist[next_node] = next_dist

                heapq.heappush(pq, (next_dist, next_node))

    return -1, float('inf'), []


def retrieve_greedy_dijkstra(g, origin, theta, max_dist, cur_res, verbal=True):
    if verbal:
        print("===========================================")
        print("Start greedy Dijkstra searching for category: ", ','.join([str(x) for x in theta]),
              " within distance ", max_dist)

    cur_node = origin
    cur_path = []
    rest_dist = max_dist

    while True:
        next_nearest_poi_node, next_nearest_dist, sub_seq = dijkstra_theta(g, cur_node, theta, rest_dist, cur_res)

        if next_nearest_dist == float('inf'):
            break

        if len(cur_path) == 0:
            cur_path = sub_seq
        else:
            cur_path = cur_path[:-1] + sub_seq

        for idx in range(CONSTANTS.CATEGORY_NUM):
            # Desire a category and node has corresponding category
            count = min(g.nodes[next_nearest_poi_node].category[idx], theta[idx])

            if count > 0:
                for each_poi in g.nodes[next_nearest_poi_node].PoIs:
                    if (each_poi.category == idx) and (each_poi.poi_id not in set([x.poi_id for x in cur_res])):
                        cur_res.add(each_poi)
                        theta[idx] -= 1

                        if theta[idx] == 0:
                            break

        if sum(theta) == 0:
            if verbal:
                print("Found an solution !!!")
                print("->".join(str(x) for x in cur_path))
                print("===========================================")

            return cur_path, max_dist - (rest_dist - next_nearest_dist), cur_res

        rest_dist -= next_nearest_dist
        cur_node = next_nearest_poi_node

    if verbal:
        print("No result")

    return None, -1, None


def greedy_process_origin(g, container_index, theta, max_dist, origin, num_origins=1, index_matrix=False,
                          verbal=True, complexity=True):
    """
    :param g: PoI network
    :param container_index: geometric container
    :param theta: categorical require [1, 2, 0, 2, 3, 1]
    :param max_dist: distance limit
    :param origin: set(), Node ID of starting query locations
    :param num_origins: preferred number of origins
    :param index_matrix: index is vector or matrix
    :param verbal: Print out information while running
    :param complexity: Return time and edge complexity
    """

    if verbal:
        print("===========================================")
        print("......Starting Origin Version......")
        print("Start searching for category: ", ','.join([str(x) for x in theta]), " within distance ", max_dist)
        print("Number of starting points: ", len(origin))

    if complexity:
        start_time = time.time()

    res_summary, res_origins = [], set()

    pq = []

    for node_id in origin:
        stock = set()

        # If there is some PoIs in query location
        if len(g.nodes[node_id].PoIs) > 0:
            satisfied_flag = 0

            for idx in range(CONSTANTS.CATEGORY_NUM):
                # Desire a category and node has corresponding category
                count = min(g.nodes[node_id].category[idx], theta[idx])

                if count > 0:
                    category_count = 0
                    for each_poi in g.nodes[node_id].PoIs:
                        if each_poi.category == idx:
                            stock.add(each_poi)
                            category_count += 1

                            if category_count >= count:  break

                # Current node has more PoIs then enough
                if count == theta[idx]:  satisfied_flag += 1

            if satisfied_flag == CONSTANTS.CATEGORY_NUM:
                if verbal:
                    print("In total ", len(res_summary) + 1, " results")

                each_res = (node_id, [node_id], 0, stock)

                if complexity:
                    each_res += (time.time() - start_time, )

                res_summary.append(each_res)

                res_origins.add(node_id)

                if len(res_origins) >= num_origins:
                    if complexity:
                        return res_summary, time.time() - start_time
                    else:
                        return res_summary

        if node_id not in res_origins:
            heapq.heappush(pq, Element(nid=node_id, res=stock))

    while pq:
        if verbal:
            print(len(pq), " elements awaiting in queue...")

        cur_element = heapq.heappop(pq)

        src_head = cur_element.find_head()

        if src_head in res_origins:
            continue

        need = [x for x in theta]

        for each_poi in cur_element.res:
            need[each_poi.category] -= 1

        if 0 < cur_element.priority <= max_dist:
            cur_res_collected = set([x for x in cur_element.res])
            cur_res_need = [x for x in need]

            rest_path, rest_dist, total_res = retrieve_greedy_dijkstra(g, cur_element.nid, cur_res_need,
                                                                       max_dist-cur_element.dist, cur_res_collected,
                                                                       verbal=False)
            #if rest_path is None:
            #    print(cur_element.dist, cur_element.priority, cur_element.print_path(), cur_element.res)
            #    exit()

            if (rest_path is not None) and (cur_element.dist + rest_dist <= max_dist):
                if verbal:
                    print("In total ", len(res_summary) + 1, " results")

                each_res = (cur_element.find_head(), cur_element.print_path() + rest_path[1:],
                            cur_element.dist + rest_dist, total_res)

                if complexity:
                    each_res += (time.time() - start_time, )

                res_summary.append(each_res)

                res_origins.add(cur_element.find_head())

                if len(res_origins) >= num_origins:
                    if complexity:
                        return res_summary, time.time() - start_time
                    else:
                        return res_summary

                continue

        if verbal:
            print("Starving for category: ", ','.join([str(x) for x in need]))

        for each_edge in g.neighbor_edges[cur_element.nid]:
            if each_edge.role == CONSTANTS.DEST or math.isnan(each_edge.weight) or each_edge.isShortcut:
                continue

            # When iterating the adjacent edges, might have new origins found
            if src_head in res_origins:
                continue

            adj_node = each_edge.node_id2

            # Check if routing a unnecessary circle - circle back to the same node without visiting any PoI
            cur_element_check = cur_element
            skip_no_PoI, loop_found = True, False

            while cur_element_check.parent:
                if g.nodes[cur_element_check.nid].PoIs:
                    skip_no_PoI = False
                    break

                if adj_node == cur_element_check.parent.nid:
                    loop_found = True
                    break

                cur_element_check = cur_element_check.parent

            if skip_no_PoI and loop_found:
                continue

            if (cur_element.nid, adj_node) not in cur_element.explored:
                if not index_matrix:
                    next_min_dist = 0
                    next_max_dist = cur_element.dist + each_edge.weight

                    for idx in range(CONSTANTS.CATEGORY_NUM):
                        if need[idx] > 0:
                            next_min_dist = max(next_min_dist, container_index[adj_node][idx])
                            next_max_dist += 2 * container_index[adj_node][idx]

                    if cur_element.dist + each_edge.weight + next_min_dist > max_dist:  continue
                else:
                    cur_collection = set([x for x in cur_element.res])

                    next_min_dist = 0

                    for idx in range(CONSTANTS.CATEGORY_NUM):
                        if next_min_dist == float('inf'):
                            break

                        if need[idx] > 0:
                            num_required_poi = need[idx]
                            satisfied_idx = 0

                            while num_required_poi > 0:
                                if container_index[adj_node][idx][satisfied_idx][0] is None:
                                    next_min_dist = float('inf')
                                    break
                                if container_index[adj_node][idx][satisfied_idx][2].poi_id not in \
                                        set([x.poi_id for x in cur_collection]):
                                    next_min_dist = max(next_min_dist, container_index[adj_node][idx][satisfied_idx][1])
                                    num_required_poi -= 1

                                satisfied_idx += 1

                    if cur_element.dist + each_edge.weight + next_min_dist > max_dist:
                        continue

                    next_max_dist = cur_element.dist + each_edge.weight
                    cur_node_id = adj_node
                    num_required_poi = [x for x in need]

                    while any(num_required_poi):
                        next_nearest_dist, picked_node = float('inf'), None

                        for idx in range(CONSTANTS.CATEGORY_NUM):
                            if num_required_poi[idx] > 0:
                                for each_poi_record in container_index[cur_node_id][idx]:
                                    if each_poi_record[0] is None:
                                        break

                                    if (each_poi_record[2].poi_id not in set([x.poi_id for x in cur_collection])) and \
                                            (each_poi_record[1] < next_nearest_dist):
                                        next_nearest_dist, picked_node = each_poi_record[1], each_poi_record
                                        break

                        if picked_node is None:
                            next_max_dist = float('inf')
                            break

                        next_max_dist += next_nearest_dist
                        cur_collection.add(picked_node[2])
                        cur_node_id = picked_node[0]
                        num_required_poi[picked_node[2].category] -= 1

                adj_res = set([x for x in cur_element.res])

                # Add necessary PoIs into .res
                # If current node does not have any PoI, then ignore
                if len(g.nodes[adj_node].PoIs) > 0:
                    for idx in range(CONSTANTS.CATEGORY_NUM):
                        # Current node has desired category
                        count = min(need[idx], g.nodes[adj_node].category[idx])

                        if count > 0:
                            for each_poi in g.nodes[adj_node].PoIs:
                                if (each_poi.category == idx) and \
                                        (each_poi.poi_id not in set([x.poi_id for x in adj_res])):
                                    adj_res.add(each_poi)

                                    count -= 1

                                    if count <= 0:
                                        break

                adj_dist = cur_element.dist + each_edge.weight

                adj_explored = set([x for x in cur_element.explored])
                adj_explored.add((cur_element.nid, adj_node))

                if not index_matrix:
                    # min: adj_dist + next_min_dist
                    # max: next_max_dist - next_min_dist
                    # 0.8 * min + 0.2 * max =
                    #   0.8 * adjDist + 0.8 * nextMinDist + 0.2 * nextMaxDist - 0.2 * NextMinDist
                    factor_priority = CONSTANTS.factorPriorVector ** sum(need)
                    adj_element = Element(nid=adj_node, priority=factor_priority * adj_dist +
                                                                 (1 - factor_priority) * next_max_dist +
                                                                 (2 * factor_priority - 1) * next_min_dist,
                                          dist=adj_dist, res=adj_res, parent=cur_element, explored=adj_explored)
                else:
                    # min: adj_dist + next_min_dist
                    # max: next_max_dist
                    # 0.5 * min + 0.5 * max
                    # factor_priority * (adj_dist + next_min_dist) + (1 - factor_priority) * next_max_dist,
                    #if next_max_dist == float('inf'):
                    #    adj_priority = next_max_dist
                    #else:
                    #    factor_priority = CONSTANTS.factorPriorMatrix ** sum(need)
                    #    adj_priority = factor_priority * (adj_dist + next_min_dist) + \
                    #                   (1 - factor_priority) * next_max_dist

                    adj_element = Element(nid=adj_node, priority=next_max_dist,
                                          dist=adj_dist, res=adj_res, parent=cur_element, explored=adj_explored)

                # Check the diversity to see if we found the results
                adj_div = [0] * CONSTANTS.CATEGORY_NUM

                for each_poi in adj_res:
                    adj_div[each_poi.category] += 1

                terminate_flag = True

                for idx in range(CONSTANTS.CATEGORY_NUM):
                    if adj_div[idx] < theta[idx]:
                        # If our collections is less than the requirement for category, then searching continues
                        terminate_flag = False
                        break

                if terminate_flag:
                    if verbal:
                        print("In total ", len(res_summary) + 1, " results")

                    each_res = (adj_element.find_head(), adj_element.print_path(), adj_dist, adj_res)

                    if complexity:
                        each_res += (time.time() - start_time, )

                    res_summary.append(each_res)

                    res_origins.add(adj_element.find_head())

                    if len(res_origins) >= num_origins:
                        if complexity:
                            return res_summary, time.time() - start_time
                        else:
                            return res_summary
                elif adj_element.find_head() not in res_origins:
                    heapq.heappush(pq, adj_element)

    if verbal:
        print("In total ", len(res_summary), " results")

    if complexity:
        return res_summary, time.time() - start_time
    else:
        return res_summary
