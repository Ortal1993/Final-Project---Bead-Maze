import numpy as np
from building_blocks import Building_Blocks
from kinematics import UR5e_PARAMS, Transform
from environment import Environment

class Layered_Graph(object):
    def __init__(self, inverse_kinematics, splines):
        self.ur_params = UR5e_PARAMS()
        self.transform = Transform(self.ur_params)
        self.env = Environment(env_idx=2)
        self.bb = Building_Blocks(self.transform, self.ur_params, self.env, inverse_kinematics)
        self.layers = []  # list of layers, each layer is a list of configurations
        self.edges = []  # list of edges, each edge is a tuple (src_node, dst_node, cost, max_dist)
        t_values = np.linspace(0, 1, 500)  # More samples for higher accuracy
        self.spline_points = np.array([spline(t_values) for spline in splines]).T

    def add_layer(self, configurations):
        layer_index = len(self.layers)
        self.layers.append(configurations)
        return layer_index

    def add_edge(self, src_node, dst_node, cost, max_dist):
        self.edges.append((src_node, dst_node, cost, max_dist))

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
                is_collision, max_dist = self.bb.local_planner(current_layer[i], current_layer[j], self.spline_points)
                if not is_collision:
                    cost = self.bb.edge_cost(current_layer[i], current_layer[j])
                    self.add_edge((layer_index, i), (layer_index, j), cost, max_dist)
                    self.add_edge((layer_index, j), (layer_index, i), cost, max_dist)
                else:
                    print("can't extend due to collision")

        # Connect nodes with the previous layer
        for i in range(len(previous_layer)):
            for j in range(len(current_layer)):
                is_collision, max_dist = self.bb.local_planner(previous_layer[i], current_layer[j], self.spline_points)
                if not is_collision:
                    cost = self.bb.edge_cost(previous_layer[i], current_layer[j])
                    self.add_edge((layer_index - 1, i), (layer_index, j), cost, max_dist)
                    self.add_edge((layer_index, j), (layer_index - 1, i), cost, max_dist)

    def build_graph(self, ik_solutions_per_layer):
        for layer_index, configurations in enumerate(ik_solutions_per_layer):
            self.add_layer(configurations)
            self.connect_layers(layer_index)
        print("Graph constructed with ", len(self.layers), " layers and ", len(self.edges), " edges")
        return self.layers, self.edges
