import Node, Edge
import CONSTANTS

import heapq, time
import math
import rtree


class ContractPoINetwork:
    def __init__(self):
        """
        nodes -- Dict{}:
            {
                key = nodeID : val = Node.Node(),
                ......
            }

        edges -- Dict{}:
            {
                key = (node_id1, node_id2) : val = Edge.Edge(),
                key = (node_id3, node_id4) : val = Edge.Edge(),
                ......
            }

        neighbor_edges -- Dict{}:
            {
                key = node_id1 : val = set{ Edge(node_id1, node_id2, weight, SRC),
                                            Edge(node_id3, node_id1, weight, DEST)
                                            ...
                                          },
                key = node_id2 : val = set{ Edge(node_id1, node_id2, weight, DEST), ... },
                ...
            }

        shortcuts -- Dict{}:
            {
                key = node_id : val = [ (cost1, (src_id1, dest_id1)),
                                        (cost2, (src_id2, dest_id2)),
                                        ...],
                ...
            }
        """
        self.nodes = {}
        self.edges = {}
        self.neighbor_edges = {}
        self.shortcuts = {}
        self.cur_iter = 0  # Current iteration => Contract Order of each node
        self.tree = None

    def add_node(self, id_, lng_, lat_, pois_=None):
        # __init__(node_id: int, lng: float, lat: float, pois=None, priority=0, isContracted=False, depth=1):
        self.nodes[id_] = Node.Node(node_id=id_, lng=lng_, lat=lat_, pois=pois_)

        if id_ not in self.neighbor_edges:
            self.neighbor_edges[id_] = set()

    def add_edge(self, id1, id2, weight):
        # __init__(node_id1, node_id2, weight, role, isShortcut=False, middle_id=-1)
        self.edges[(id1, id2)] = Edge.Edge(node_id1=id1, node_id2=id2, weight=weight, role=CONSTANTS.SRC)

        self.neighbor_edges[id1].add(Edge.Edge(node_id1=id1, node_id2=id2, weight=weight, role=CONSTANTS.SRC))
        self.neighbor_edges[id2].add(Edge.Edge(node_id1=id1, node_id2=id2, weight=weight, role=CONSTANTS.DEST))

    def add_shortcut(self, id1, id2, weight, mid_id):
        # __init__(node_id1, node_id2, weight, role, isShortcut=False, middle_id=-1)
        # id1 -> mid_id -> id2
        self.edges[(id1, id2)] = Edge.Edge(node_id1=id1, node_id2=id2, weight=weight, role=CONSTANTS.SRC,
                                           isShortcut=True, middle_id=mid_id)

        self.neighbor_edges[id1].add(Edge.Edge(node_id1=id1, node_id2=id2, weight=weight, role=CONSTANTS.SRC,
                                               isShortcut=True, middle_id=mid_id))
        self.neighbor_edges[id2].add(Edge.Edge(node_id1=id1, node_id2=id2, weight=weight, role=CONSTANTS.DEST,
                                               isShortcut=True, middle_id=mid_id))

    def update_depth(self, node_id):
        # Check each adjacent vertex of node_id
        for eachE in self.neighbor_edges[node_id]:
            if (eachE.node_id1 != node_id) and (not self.nodes[eachE.node_id1].isContracted):
                self.nodes[eachE.node_id1].depth = max(self.nodes[eachE.node_id1].depth,
                                                       self.nodes[node_id].depth + 1)

            if (eachE.node_id2 != node_id) and (not self.nodes[eachE.node_id2].isContracted):
                self.nodes[eachE.node_id2].depth = max(self.nodes[eachE.node_id2].depth,
                                                       self.nodes[node_id].depth + 1)

    def number_contracted(self, edges):
        count = 0

        for eachE in edges:
            if eachE.role == CONSTANTS.SRC:
                dest = self.nodes[eachE.node_id2]
                if dest.isContracted:  count += 1
            else:
                src = self.nodes[eachE.node_id1]
                if src.isContracted:  count += 1

        return count

    def shortcut_required_from_src(self, src, mid_id, dests, distance_func=None):
        pq, seen, min_dist = [(0, src, tuple())], set(), {src: 0}

        res = {}

        while pq:
            cur_dist, cur_node, path = heapq.heappop(pq)

            if (cur_node in seen) or self.nodes[cur_node].isContracted:
                # If the current node has been visited or has been contracted and thus removed
                continue

            seen.add(cur_node)
            path += (cur_node,)

            if cur_node in dests:
                res[cur_node] = (cur_dist, path)

            if all([(x in seen) for x in dests]):
                break

            for nextE in self.neighbor_edges[cur_node]:
                # nextE: curNode --  curNode -> nextNode
                if nextE.role == CONSTANTS.SRC:
                    w, next_node = nextE.weight, nextE.node_id2

                    if (next_node in seen) or self.nodes[next_node].isContracted:
                        continue

                    prev_dist = min_dist.get(next_node, None)
                    next_dist = cur_dist + w

                    if (prev_dist is None) or (next_dist < prev_dist):
                        min_dist[next_node] = next_dist
                        heapq.heappush(pq, (next_dist, next_node, path))

        shortcuts_list = []
        weighted_sum = 0

        for each_dest in res:
            if res[each_dest][1] == (src, mid_id, each_dest):
                shortcuts_list.append((res[each_dest][0], (src, each_dest)))

                if distance_func is "cosine":
                    cos_sum = 0
                    len1, len2 = 0, 0

                    for idx in range(CONSTANTS.CATEGORY_NUM):
                        cos_sum += self.nodes[src].category[idx] * self.nodes[each_dest].category[idx]
                        len1 += self.nodes[src].category[idx] ** 2
                        len2 += self.nodes[each_dest].category[idx] ** 2

                    weighted_sum += 1 + 1 - cos_sum / (math.sqrt(len1) * math.sqrt(len2))
                elif distance_func is "jaccard":
                    min_sum, max_sum = 0, 0

                    for idx in range(CONSTANTS.CATEGORY_NUM):
                        min_sum += min(self.nodes[src].category[idx], self.nodes[each_dest].category[idx])
                        max_sum += max(self.nodes[src].category[idx], self.nodes[each_dest].category[idx])

                    if max_sum != 0:
                        jc_sim = min_sum / max_sum
                    else:
                        jc_sim = 1

                    weighted_sum += 1 + 1 - jc_sim
                else:
                    square_sum = 0

                    for idx in range(CONSTANTS.CATEGORY_NUM):
                        square_sum += (self.nodes[src].category[idx] - self.nodes[each_dest].category[idx]) ** 2

                    weighted_sum += 1 + math.sqrt(square_sum)

        return shortcuts_list, weighted_sum

    def shortcuts_needed_per_node(self, mid_id):
        # src -> mid_id -> dest
        print("Starting computing the shortcuts of Node ", mid_id)

        srcs, dests = set(), set()

        for eachE in self.neighbor_edges[mid_id]:
            if (eachE.role == CONSTANTS.SRC) and (not self.nodes[eachE.node_id2].isContracted):
                # midID -> node_id2
                # If a vertex has been contracted, all its adjacent edges will be removed
                dests.add(eachE.node_id2)
            elif (eachE.role == CONSTANTS.DEST) and (not self.nodes[eachE.node_id1].isContracted):
                # node_id1 -> midID
                srcs.add(eachE.node_id1)

        shortcuts_list = []
        weighted_sum = 0

        for src in srcs:
            # [(cost1, (src, dest1)), (cost2, (src, dest2)), ...]
            each_shortcuts_list, each_weighted_sum = self.shortcut_required_from_src(src, mid_id, dests)

            shortcuts_list.extend(each_shortcuts_list)
            weighted_sum += each_weighted_sum

        print("Done with the shortcuts of Node ", mid_id)

        return shortcuts_list, weighted_sum

    def priority_cal(self, node_id):
        """
        Edge Quotient: #ofShortcutsAdded / #ofEdgesRemovedFromRemainNetwork
                       where -
                            #ofEdgesRemovedFromRemainNetwork = #ofNeighbors - #ofNeighborsContracted (dismiss direction)
                       range: 0 ~ n-1 where n = #ofNeighbors

        Hierarchy Depth: n.depth
        """
        print("Starting calculating priority of Node ", node_id)

        # Numbers fo Edges removed from remain network
        adjacent_edges = self.neighbor_edges[node_id]
        num_removed_edges = len(adjacent_edges) - self.number_contracted(adjacent_edges)

        if num_removed_edges > 0:
            # Found necessary shortcuts
            shortcuts_list, weighted_sum = self.shortcuts_needed_per_node(node_id)
            edge_quotient = weighted_sum / num_removed_edges
        else:
            # all neighbors has been contracted
            shortcuts_list, weighted_sum = [], 0
            edge_quotient = 0

        priority_res = 2 * edge_quotient + self.nodes[node_id].depth

        print("Done with the priority of Node ", node_id)
        print("Number of shortcuts: ", len(shortcuts_list), '\n', shortcuts_list)
        print("Weighted sum of shortcuts: ", weighted_sum)
        print("Number of edges removed: ", num_removed_edges)
        print("Priority: ", priority_res)
        print("====================================================================")

        return node_id, priority_res, shortcuts_list

    def check_independent(self, node_id, confirmed_independent):
        # If confirmed_independent[.] is False, then it must be modified to False by other process
        if not confirmed_independent[node_id]:
            print("Node ", node_id, " has already been determined as NOT independent by other process !!!")
            return node_id, False

        adjacent_edges = self.neighbor_edges[node_id]
        neighbor_nodes = set()

        # 1-hop neighborhoods
        for eachE in adjacent_edges:
            if (eachE.node_id1 != node_id) and (not self.nodes[eachE.node_id1].isContracted):
                neighbor_nodes.add(eachE.node_id1)
            if (eachE.node_id2 != node_id) and (not self.nodes[eachE.node_id2].isContracted):
                neighbor_nodes.add(eachE.node_id2)

        # 2-hop neighborhoods
        neighbors_2hop_nodes = set()
        for neighbor_1hop_node in neighbor_nodes:
            adjacent_2hop_edges = self.neighbor_edges[neighbor_1hop_node]
            for eachE in adjacent_2hop_edges:
                if (eachE.node_id1 != node_id) and (not self.nodes[eachE.node_id1].isContracted):
                    neighbors_2hop_nodes.add(eachE.node_id1)
                if (eachE.node_id2 != node_id) and (not self.nodes[eachE.node_id2].isContracted):
                    neighbors_2hop_nodes.add(eachE.node_id2)

        # 1-hop and 2-hop
        neighbor_nodes |= neighbors_2hop_nodes

        for each_neighbor_node in neighbor_nodes:
            if self.nodes[each_neighbor_node].priority < self.nodes[node_id].priority:
                # current node does NOT obtain the smallest priority
                confirmed_independent[node_id] = False
                print("Node ", node_id, " is NOT independent...")
                return node_id, False
            elif (self.nodes[each_neighbor_node].priority == self.nodes[node_id].priority) and \
                    (each_neighbor_node < node_id):
                # Tir breaker: pick the smaller id if the priority is the same
                confirmed_independent[node_id] = False
                print("Node ", node_id, " is NOT independent...")
                return node_id, False

        print("Node ", node_id, " is independent...")

        # Thus all its neighbors cannot be independent
        for each_node in neighbor_nodes:
            confirmed_independent[each_node] = False

        return node_id, True

    def vector_container_cal(self, node_id):
        """
        :param node_id: Integer, ID of source
        :return: res: [d1, d2, ..., dC]
            where distX is the shortest distance of nID -> ... -> category Y
        """
        print("Building Vector-Container for Node ", node_id, "......")

        start_time = time.time()

        res = [float('inf')] * CONSTANTS.CATEGORY_NUM

        # Shortest distance to each node
        dist = [float('inf')] * len(self.nodes)
        dist[node_id] = 0

        pq = []

        ### (distance, node ID, 0 for up or 1 for down)
        heapq.heappush(pq, (0, node_id, 0))

        # Instead of decrease priority in pq, we record visited nodes
        # Decrease priority needs O(n) in worst case by using binary tree, checking visited set uses constant time
        visited = set()

        while pq:
            cur_dist, cur_node, cur_stall = heapq.heappop(pq)

            # If node has been explored, then skip
            if cur_node in visited:
                continue

            visited.add(cur_node)

            # If current node has no PoI, then no need to update
            if any(self.nodes[cur_node].category):
                # print("Found PoIs on Node ", curNode, " for Source ", nID)
                for idx, val in enumerate(self.nodes[cur_node].category):
                    if val != 0 and res[idx] == float('inf'):
                        res[idx] = cur_dist

                # If found all nearest PoIs in each category (i.e., no inf in vector), then stop
                if float('inf') not in res:
                    print("Done with Node ", node_id, "/", len(self.nodes), " by using ", time.time() - start_time,
                          "s......")
                    if cur_node == node_id:  print("***END AT START***")
                    print(res)
                    print('===================================================')
                    return node_id, res

            for eachE in self.neighbor_edges[cur_node]:
                # eachE: nodeID -> nodeID2
                if eachE.role == CONSTANTS.SRC:
                    adjacent_node = eachE.node_id2

                    # 0 for up or 1 for down
                    next_stall = 0

                    if self.nodes[adjacent_node].contract_order < self.nodes[cur_node].contract_order:  next_stall = 1

                    if cur_stall == 1 and next_stall == 0:  # Down-then-Up is not allowed
                        continue
                    else:
                        next_dist = cur_dist + eachE.weight

                        if next_dist < dist[adjacent_node]:
                            heapq.heappush(pq, (next_dist, adjacent_node, next_stall))
                            dist[adjacent_node] = next_dist

        print("Done with Node ", node_id, "/", len(self.nodes), " by using ", time.time() - start_time, "s......")
        print(res)
        print('===================================================')

        return node_id, res

    def matrix_container_cal(self, node_id):
        """
        :param node_id: Integer, ID of source
        :return: res: [
                        [(v11, d11, PoI11), (v12, d12, PoI12), ..., (v1K, d1K, PoI1K)],   -- category 1
                        [(v21, d21, PoI21), (v22, d22, PoI22), ..., (v2K, d2K, PoI2K)],   -- category 2
                        ...
                        [(vC1, dC1, PoIC1), (vC2, dC2, PoIC2), ..., (vCK, dCK, PoICK)],   -- category C
                      ]
            where distXY is the Y shortest distance of nID -> ... -> category X,
                  vXY is the corresponding vertex id,
                  PoIXY is the corresponding PoI
                  (Note that, e.g., v12 and v13 might be the same if there are 2 PoIs belonged to category 1
                  on same vertex, but PoI12 and PoI13 would be different)
        """

        #print("Building Matrix-Container for Node ", node_id, "......")

        start_time = time.time()

        res = [[(None, float('inf'), None)] * CONSTANTS.K_MAX for _ in range(CONSTANTS.CATEGORY_NUM)]

        # How many POIs in each category have been found -- index from 0
        found_poi_per_category = [0] * CONSTANTS.CATEGORY_NUM

        # Shortest distance to each node
        dist = [float('inf')] * len(self.nodes)
        dist[node_id] = 0

        pq = []

        # (distance, node ID, 0 for up or 1 for down)
        #heapq.heappush(pq, (0, node_id, 0))
        heapq.heappush(pq, (0, node_id))

        # Instead of decrease priority in pq, we record visited nodes
        # Decrease priority needs O(n) in worst case by using binary tree, checking visited set uses constant time
        visited = set()

        while pq:
            #cur_dist, cur_node, cur_stall = heapq.heappop(pq)
            cur_dist, cur_node = heapq.heappop(pq)

            # If node has been explored, then skip
            if cur_node in visited:
                continue

            visited.add(cur_node)

            # If current node has no PoI, then no need to update
            if any(self.nodes[cur_node].category):
                for idx, val in enumerate(self.nodes[cur_node].category):
                    if val > 0 and found_poi_per_category[idx] < CONSTANTS.K_MAX:
                        for each_poi in self.nodes[cur_node].PoIs:
                            if each_poi.category == idx:
                                res[idx][found_poi_per_category[idx]] = (cur_node, cur_dist, each_poi)
                                found_poi_per_category[idx] += 1

                                if found_poi_per_category[idx] >= CONSTANTS.K_MAX:
                                    break

                if all([True if each_row[CONSTANTS.K_MAX - 1][1] != float('inf') else False for each_row in res]):
                    #print("Done with Node ", node_id, "/", len(self.nodes), " by using ", time.time() - start_time,
                    #      "s......")
                    #print("Collected all categories")
                    if cur_node == node_id:
                        print("***END AT START***")
                    #print(res)
                    #print('===================================================')
                    return node_id, res, time.time() - start_time

            for eachE in self.neighbor_edges[cur_node]:
                # eachE: nodeID -> nodeID2
                if eachE.role == CONSTANTS.SRC:
                    adjacent_node = eachE.node_id2

                    # 0 for up or 1 for down
                    '''
                    next_stall = 0

                    if self.nodes[adjacent_node].contract_order < self.nodes[cur_node].contract_order:  next_stall = 1

                    if cur_stall == 1 and next_stall == 0:
                        # Down-then-Up would not lead to the shortest path
                        continue
                    else:
                        next_dist = cur_dist + eachE.weight

                        if next_dist < dist[adjacent_node]:
                            heapq.heappush(pq, (next_dist, adjacent_node, next_stall))
                            dist[adjacent_node] = next_dist
                    '''
                    next_dist = cur_dist + eachE.weight

                    if next_dist < dist[adjacent_node]:
                        heapq.heappush(pq, (next_dist, adjacent_node))
                        dist[adjacent_node] = next_dist

        #print("Done with Node ", node_id, " by using ", time.time() - start_time, "s......")
        #print(res)
        #print('===================================================')

        return node_id, res, time.time() - start_time

    def bulk_loading_generator(self, node_list):
        for i, node_id in enumerate(node_list):
            (lng, lat) = (self.nodes[node_id].lng, self.nodes[node_id].lat)
            yield (i, (lng, lat, lng, lat), node_id)

    def rtree_build(self, node_list):
        print("========================================")
        print("Start building rtree...")

        node_list = list(node_list)

        self.tree = rtree.index.Index(self.bulk_loading_generator(node_list))

        print("Done with rtree from ", len(node_list), " origins......")
        print("========================================")

    def multi_origins_dijkstra(self, sources, target, budget=float('inf')):
        """
        :param sources: set of node IDs
        :param target: Integer, node ID
        :param budget: Integer, distance
        """

        if target in sources:
            return [target], 0

        dists = [float('inf')] * len(self.nodes)
        pq = []
        path = {}
        visited = set()

        for each_src in sources:
            heapq.heappush(pq, (0, each_src))
            path[each_src] = None
            dists[each_src] = 0

        while pq:
            cur_dist, cur_node = heapq.heappop(pq)

            if cur_dist > budget:  return [], float('inf')

            if cur_node == target:
                res_path = []

                reverse_head = cur_node

                while reverse_head is not None:
                    res_path.append(reverse_head)
                    reverse_head = path[reverse_head]

                res_path.reverse()

                return res_path, cur_dist

            if cur_node in visited:  continue

            visited.add(cur_node)

            for each_edge in self.neighbor_edges[cur_node]:
                if each_edge.role == CONSTANTS.DEST or math.isnan(each_edge.weight):  continue

                next_dist = cur_dist + each_edge.weight

                if next_dist > budget:  continue

                next_node = each_edge.node_id2

                '''
                if next_node == target:
                    res_path = [next_node]

                    reverse_head = cur_node

                    while reverse_head is not None:
                        res_path.append(reverse_head)
                        reverse_head = path[reverse_head]

                    res_path.reverse()

                    return res_path, next_dist
                '''

                if next_dist < dists[next_node]:
                    path[next_node] = cur_node
                    dists[next_node] = next_dist

                    heapq.heappush(pq, (next_dist, next_node))

        return [], float('inf')

    def multi_origins_single_target_reverse_dijkstra(self, origins, target, budget=float('inf'),
                                                     num_origins_need=1, picked_origins=None):
        """
        :param origins: set(Integer, node ID)
        :param target: Integer, node ID
        :param budget: Integer, distance
        :param num_origins_need: Integer, Number of sources needed
        :param picked_origins: set(Integer, node ID), the origins already selected - we need something else not in set
        """

        # Apply Dijkstra from single target to all sources
        # Reverse all edges
        # Terminate if dist > budget OR > k sources founded

        if picked_origins is None:  picked_origins = set()

        res = []
        new_picked_origins = set()

        dists = [float('inf')] * len(self.nodes)
        pq = []
        path = {}
        visited = set()

        heapq.heappush(pq, (0, target))
        path[target] = None
        dists[target] = 0

        while pq:
            cur_dist, cur_node = heapq.heappop(pq)

            if cur_dist > budget:
                return res

            if cur_node in visited:
                continue

            if (cur_node in origins) and (cur_node not in picked_origins) and (cur_node not in new_picked_origins):
                new_picked_origins.add(cur_node)

                res_path = []

                reverse_head = cur_node

                while reverse_head is not None:
                    res_path.append(reverse_head)
                    reverse_head = path[reverse_head]

                found_res = (res_path, cur_dist)

                res.append(found_res)

                if len(res) >= num_origins_need:
                    return res

                if len(new_picked_origins) + len(picked_origins) == len(origins):
                    return res

            visited.add(cur_node)

            for each_edge in self.neighbor_edges[cur_node]:
                # Reverse the edge, so we want cur_node == each_edge.DEST
                if each_edge.role == CONSTANTS.SRC or math.isnan(each_edge.weight):
                    continue

                next_dist = cur_dist + each_edge.weight

                if next_dist > budget:
                    continue

                prev_node = each_edge.node_id1

                if next_dist < dists[prev_node]:
                    path[prev_node] = cur_node
                    dists[prev_node] = next_dist

                    heapq.heappush(pq, (next_dist, prev_node))

        return res

    def bi_dijkstra(self, source, target, budget=float('inf')):
        """
        :param source: set of node IDs
        :param target: Integer, node ID
        :param budget: Integer, distance
        """

        if target in source:
            return [target], 0

        # init: [Forward, Backward]
        dists = [{}, {}]  # Real shortest distance to each node
        fringe = [[], []]  # Priority Queue for both directions
        seen = [{}, {}]  # Current "optimal" distance to seen nodes
        paths = [{}, {}]  # Path tree

        for each_src in source:
            heapq.heappush(fringe[0], (0, each_src))
            seen[0][each_src] = 0
            paths[0][each_src] = [each_src]

        heapq.heappush(fringe[1], (0, target))
        seen[1][target] = 0
        paths[1][target] = [target]

        # 0 for Forward, 1 for Backward
        direction = 1

        direct_check = [CONSTANTS.SRC, CONSTANTS.DEST]

        res_path, res_dist = [], float('inf')

        while fringe[0] and fringe[1]:
            direction = 1 - direction

            cur_dist, cur_node = heapq.heappop(fringe[direction])

            if cur_dist > budget and fringe[1 - direction][0][0] > budget:  break

            if cur_node in dists[direction]:
                # Check if the sum of the smallest distance from two pq has exceeded the budget and/or current result
                # Note that the current smallest sum has no need to be a path

                ### if fringe[1 - direction] and fringe[1 - direction][0][1] in dists[1 - direction] and \
                ###        ((dists[direction][cur_node] + dists[1 - direction][fringe[1 - direction][0][1]] > budget) or
                ###         (dists[direction][cur_node] + dists[1 - direction][fringe[1 - direction][0][1]] >= res_dist)):
                ###    break

                # Has found earlier
                continue

            dists[direction][cur_node] = cur_dist

            # It is ensured to be correct because we check seen[0] + seen[1] in the loop below
            if cur_node in dists[1 - direction]:  return res_path, res_dist

            # stall-on-demand
            stall_on_demand = False

            # Upward:   we get a node v - if the downward direction (w, v) with d(src, w) + weight(w, v) < d(src, v),
            #           then v can be ignored because src -> ... -> w -> v  is shorter
            # Downward: we get a node v - if the upward   direction (v, w) with weight(v, w) + d(w, dest) < d(v, dest),
            #           then v can be ignored because v -> w -> ... -> dest is shorter
            for each_edge in self.neighbor_edges[cur_node]:
                if math.isnan(each_edge.weight):  continue

                if direction == 0 and each_edge.role == CONSTANTS.DEST:
                    income_node = each_edge.node_id1
                elif direction == 1 and each_edge.role == CONSTANTS.SRC:
                    income_node = each_edge.node_id2
                else:
                    continue

                if self.nodes[income_node].contract_order >= self.nodes[cur_node].contract_order:
                    if income_node in seen[direction] and seen[direction][income_node] + each_edge.weight < cur_dist:
                        stall_on_demand = True
                        break

            if stall_on_demand:  continue

            for each_edge in self.neighbor_edges[cur_node]:
                if math.isnan(each_edge.weight):  continue

                # Forward  search in Upward   graph,  we want eachE.role == SRC  :
                #               cur_node -> node_id2 -AND- order(cur_node) < order(node_id2)
                # Backward search in Downward graph,  we want eachE.role == DEST :
                #               node_id1 -> cur_node -AND- order(node_id1) > order(cur_node)
                # Otherwise, skip current iteration
                if each_edge.role == direct_check[1 - direction]:  continue

                if direction == 0:
                    adjacent_node = each_edge.node_id2
                else:
                    adjacent_node = each_edge.node_id1

                if self.nodes[cur_node].contract_order > self.nodes[adjacent_node].contract_order:  continue

                next_dist = cur_dist + each_edge.weight

                if next_dist > budget:  continue

                if adjacent_node in dists[direction]:
                    continue
                elif adjacent_node not in seen[direction] or next_dist < seen[direction][adjacent_node]:
                    seen[direction][adjacent_node] = next_dist
                    heapq.heappush(fringe[direction], (next_dist, adjacent_node))
                    paths[direction][adjacent_node] = paths[direction][cur_node] + [adjacent_node]

                    if adjacent_node in seen[1-direction]:
                        total_dist = next_dist + seen[1-direction][adjacent_node]

                        if total_dist < res_dist:
                            res_dist = total_dist
                            rev_path = [x for x in paths[1][adjacent_node]]
                            rev_path.reverse()
                            res_path = paths[0][adjacent_node] + rev_path[1:]

                            if res_dist <= budget:  return res_path, res_dist

        return res_path, res_dist
