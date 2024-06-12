import numpy as np
import heapq

class Dijkstra(object):
    def __init__(self, graph, start, end):
        self.graph = graph #list of layers - each layer is a list of configurations
                           #list of edges - each edge is a tuple (src_node, dst_node, cost, max_dist)
        self.start = start
        self.end = end

    def find_shortest_path(self):
        """
        Find the bottleneck cost from start to target in the given graph.
        """

        # Dictionary to store the bottleneck cost for each node
        b_neck = {self.start: 0}
        predecessors = {self.start: None}
        
        # Priority queue (min-heap) for nodes to explore
        open_set = []
        # Initialize the open set with the start node
        #the heap property based on the first element of the tuples pushed into it
        heapq.heappush(open_set, (b_neck[self.start], self.start))  # (node, bottleneck cost)
        
        while open_set:
            # Get the node with the minimum bottleneck cost
            current_bottleneck, v_curr = heapq.heappop(open_set)
            
            # If we reached the target, return the bottleneck cost
            if v_curr == self.end:
                path = self.compute_path(v_curr, predecessors) 
                return current_bottleneck, np.array(path)
            
            if v_curr in self.graph.edges.keys():
                neighbors = self.graph.get_neighbors(v_curr)
            else:
                continue
            # Explore the neighbors
            for neighbor in neighbors:
                (v1, v2), edge_cost, dist, node = neighbor
                new_bottleneck = max(edge_cost, current_bottleneck)
                if any((v1, v2) == v for _, v in open_set):
                    if b_neck[(v1, v2)] < new_bottleneck:
                        continue
                    else:
                        b_neck[(v1, v2)] = new_bottleneck
                        predecessors[(v1, v2)] = v_curr
                        heapq.heappush(open_set, (new_bottleneck, (v1, v2)))
        
        # If the target is not reachable
        return None, []    

    def compute_path(self, v, predecessors):
        path = []
        while v is not None:
            path.append(v)
            v = predecessors[v]
        path.reverse()
        return path