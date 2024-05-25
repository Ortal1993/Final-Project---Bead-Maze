import numpy as np
from building_blocks import Building_Blocks
from kinematics import UR5e_PARAMS, Transform
from environment import Environment

class Layered_Graph(object):
    def __init__(self):
        self.ur_params = UR5e_PARAMS()
        self.transform = Transform(self.ur_params)
        self.env = Environment(env_idx=2) #TODO - I think it suppose to be 2, it was 3 before, or maybe we can carete anv_index 4 with our environment.
        self.bb = Building_Blocks(self.transform, self.ur_params, self.env)
        self.layers = []  # list of layers, each layer is a list of configurations
        self.edges = []   # list of edges, each edge is a tuple (src_node, dst_node, cost)

    def add_layer(self, configurations):
        layer_index = len(self.layers)
        self.layers.append(configurations)
        return layer_index

    def add_edge(self, src_node, dst_node, cost):
        self.edges.append((src_node, dst_node, cost))

    """def add_node(self, layer_index, configuration):
        if layer_index >= len(self.layers):
            self.layers.append([])
        self.layers[layer_index].append(configuration)
        node_index = len(self.layers[layer_index]) - 1
        self.edges[(layer_index, node_index)] = []"""
    
    def get_nodes(self, layer_index):
        return self.layers[layer_index]

    def get_edges(self, from_node):
        return self.edges[from_node]

    def connect_layers(self, layer_index):
        if layer_index == 0:
            return  # No previous layer to connect to

        current_layer = self.layers[layer_index]
        previous_layer = self.layers[layer_index - 1]

        # Connect nodes within the same layer
        for i in range(len(current_layer)):
            for j in range(i + 1, len(current_layer)):
                if self.bb.local_planner(current_layer[i], current_layer[j]):
                    cost = self.bb.edge_cost(current_layer[i], current_layer[j])
                    self.add_edge((layer_index, i), (layer_index, j), cost)
                    self.add_edge((layer_index, j), (layer_index, i), cost)
                else:
                    print("can't extend due to collision")

        # Connect nodes with the previous layer
        for i in range(len(previous_layer)):
            for j in range(len(current_layer)):
                if self.bb.local_planner(previous_layer[i], current_layer[j]):
                    cost = self.bb.edge_cost(previous_layer[i], current_layer[j])
                    self.add_edge((layer_index - 1, i), (layer_index, j), cost)
                    self.add_edge((layer_index, j), (layer_index - 1, i), cost)

    def build_graph(self, ik_solutions_per_layer):
        for layer_index, configurations in enumerate(ik_solutions_per_layer):
            self.add_layer(configurations)
            self.connect_layers(self.bb, layer_index)
        print("Graph constructed with ", len(self.layers), " layers and ", len(self.edges), " edges")
        return self.layers, self.edges
