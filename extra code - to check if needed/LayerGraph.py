import numpy as np
from building_blocks import Building_Blocks
from kinematics import UR5e_PARAMS, Transform
from environment import Environment


class Layer_Graph(object):
    def __init__(self):
        self.layers = []
        self.edges = {}

    def add_node(self, layer_index, configuration):
        if layer_index >= len(self.layers):
            self.layers.append([])
        self.layers[layer_index].append(configuration)
        node_index = len(self.layers[layer_index]) - 1
        self.edges[(layer_index, node_index)] = []

    def add_edge(self, from_node, to_node):
        self.edges[from_node].append(to_node)

    def get_nodes(self, layer_index):
        return self.layers[layer_index]

    def get_edges(self, from_node):
        return self.edges[from_node]


def populate_graph(graph, bb, layer_index, configurations, previous_layer_configs):
    for config in configurations:
        graph.add_node(layer_index, config)

    # Connect within the same layer
    if len(graph.get_nodes(layer_index)) > 1:
        for i in range(len(graph.get_nodes(layer_index))):
            for j in range(i + 1, len(graph.get_nodes(layer_index))):
                if bb.local_planner(graph.get_nodes(layer_index)[i], graph.get_nodes(layer_index)[j]):
                    graph.add_edge((layer_index, i), (layer_index, j))
                    graph.add_edge((layer_index, j), (layer_index, i))

    # Connect with the previous layer
    if previous_layer_configs is not None and layer_index > 0:
        for i, prev_config in enumerate(previous_layer_configs):
            for j, config in enumerate(graph.get_nodes(layer_index)):
                if bb.local_planner(prev_config, config):
                    graph.add_edge((layer_index - 1, i), (layer_index, j))
                    graph.add_edge((layer_index, j), (layer_index - 1, i))


def main():
    ur_params = UR5e_PARAMS()
    transform = Transform(ur_params)
    env = Environment(env_idx=3)
    bb = Building_Blocks(transform, ur_params, env)

    graph = Layer_Graph()
    num_waypoints = 5  # Total number of waypoints
    for i in range(num_waypoints):
        configurations = [np.random.rand(6) for _ in range(3)]  # Example configurations
        previous_configs = graph.get_nodes(i - 1) if i > 0 else None
        populate_graph(graph, bb, i, configurations, previous_configs)

    # Here you can now use the graph for planning or analysis
    a=o


if __name__ == "__main__":
    main()
