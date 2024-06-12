import numpy as np
#from building_blocks import Building_Blocks
from kinematics import Transform
#from environment import Environment

class Layered_Graph(object):
    def __init__(self, ur_params, env, bb, splines):
        self.ur_params = ur_params
        self.transform = Transform(self.ur_params)
        self.env = env
        self.bb = bb
        self.layers = []  # list of layers, each layer is a list of configurations
        self.edges = {}  # dict of edges, key is the src_node(tuple), value is tuple of (dst_node, cost, max_dist)
        t_values = np.linspace(0, 1, 500)  # More samples for higher accuracy
        self.spline_points = np.array([spline(t_values) for spline in splines]).T

    def add_layer(self, configurations):
        self.layers.append(configurations)

    def add_edge(self, src_node, dst_node, cost, max_dist, node):
        if src_node in self.edges.keys():
            self.edges[src_node] += [[dst_node, cost, max_dist, node]]
        else:
            self.edges[src_node] = [[dst_node, cost, max_dist, node]]

    def get_nodes_by_layer(self, layer_index):
        return self.layers[layer_index]

    def get_edges(self, src_node):
        return self.edges.get(src_node)

    def connect_layers(self, layer_index):
        current_layer_nodes = self.layers[layer_index]
        
        # Connect nodes within the same layer
        for i in range(len(current_layer_nodes)):
            for j in range(i + 1, len(current_layer_nodes)):
                is_collision, max_dist = self.bb.local_planner(current_layer_nodes[i], 
                                                               current_layer_nodes[j], 
                                                               self.spline_points)
                if not is_collision:
                    v_i = current_layer_nodes[i]
                    vertex_i = [item for sublist in v_i for item in sublist]

                    v_j = current_layer_nodes[j]
                    vertex_j = [item for sublist in v_j for item in sublist]

                    cost = self.bb.edge_cost(v_i, v_j)
                    self.add_edge((layer_index, i), (layer_index, j), cost, max_dist, (vertex_i, vertex_j))
                    self.add_edge((layer_index, j), (layer_index, i), cost, max_dist, (vertex_j, vertex_i))
                    print("extend")
                else:
                    print("can't extend due to collision")

        # No previous layer to connect to 
        if layer_index == 0:
            return
        previous_layer_nodes = self.layers[layer_index - 1]

        # Connect nodes with the previous layer
        for i in range(len(previous_layer_nodes)):
            for j in range(len(current_layer_nodes)):
                is_collision, max_dist = self.bb.local_planner(previous_layer_nodes[i], 
                                                               current_layer_nodes[j], 
                                                               self.spline_points)
                if not is_collision:
                    v_i = previous_layer_nodes[i]
                    vertex_i = [item for sublist in v_i for item in sublist]

                    v_j = current_layer_nodes[j]
                    vertex_j = [item for sublist in v_j for item in sublist]

                    cost = self.bb.edge_cost(v_i, v_j)
                    self.add_edge((layer_index - 1, i), (layer_index, j), cost, max_dist, (vertex_i, vertex_j))
                    self.add_edge((layer_index, j), (layer_index - 1, i), cost, max_dist, (vertex_i, vertex_j))
                    print("extend with previous")
                else:
                    print("can't extend with previous due to collision")

    def build_graph(self, ik_solutions_per_layer):
        layer_index = 0
        for configurations in ik_solutions_per_layer:
            if len(configurations) != 0:
                self.add_layer(configurations)
                self.connect_layers(layer_index)
                layer_index += 1
        print("Graph constructed with ", len(self.layers), " layers and ", len(self.edges), " edges")

    def get_neighbors(self, v):
        return self.get_edges(v)


