import numpy as np
import argparse
import os
from inverse_kinematics import Inverse_Kinematics
from building_blocks import Building_Blocks
from kinematics import UR5e_PARAMS, Transform
from environment import Environment
from visualizer import Visualize_UR
from visualizer_gif import Visualize_Gif

from layered_graph import Layered_Graph
from path_model import Path_Model
from dijkstra import Dijkstra

import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='script for bead maze')
    parser.add_argument('-waypoints', '--waypoints', type=str, 
                        default='way_points.json',
                        help='Json file name containing all way points')
    args = parser.parse_args()

    path_model = Path_Model(json_file=args.waypoints)
    splines, smooth_path, tangents = path_model.process_path()
    waypoints_coords = path_model.get_waypoints_coords()

    # Initialize the inverse kinematics solver with initial parameters
    tx, ty, tz = waypoints_coords[0]
    ik_solver = Inverse_Kinematics(tx, ty, tz, tangents[0])

    ur_params = UR5e_PARAMS()
    transform = Transform(ur_params)
    env = Environment(env_idx=2)
    bb = Building_Blocks(transform, ur_params, env, ik_solver)
    
    home = np.deg2rad([0, -90, 0, -90, 0,0 ])
    visualizer = Visualize_UR(ur_params, env=env, 
                              transform=transform, bb=bb)
    visualizer.show_our_path(smooth_path, tangents, home, 0.1)

    ik_solutions_per_layer = []
    # Process each waypoint and its corresponding tangent
    for waypoint, tangent in zip(smooth_path, tangents):
        tx, ty, tz = waypoint
        # Update IK solver target and tangent
        ik_solver.update_target_and_tangent(tx, ty, tz, tangent)

        # Compute IK solutions for the current waypoint
        possible_configs = ik_solver.find_possible_configs()
        possible_configs_flatten = np.array([np.squeeze(sub_array) for sub_array in possible_configs])

        ik_solutions_per_layer.append(possible_configs_flatten)#TODO - verify that it is list of list
    
    # Building the graph using valid configurations for each waypoint
    graph = Layered_Graph(ur_params, env, bb, splines)
    graph.build_graph(ik_solutions_per_layer)

    first_layer = 0
    last_layer = len(graph.layers)
    nodes_in_first_layer = graph.get_nodes_by_layer(first_layer)    
    nodes_in_last_layer = graph.get_nodes_by_layer(last_layer - 1)
    min_bottle_neck = np.inf
    shortest_path = None
    
    for i in range(len(nodes_in_first_layer)):
        dijkstra = Dijkstra(graph, (first_layer, i), last_layer - 1)
        curr_bottleneck, path = dijkstra.find_shortest_path()
        if curr_bottleneck != None:
            if curr_bottleneck < min_bottle_neck:
                min_bottle_neck = curr_bottleneck
                shortest_path = path
                print("shortest path:\n")
                print(shortest_path)
            else:
                print("Did not found a new shortest path")
            

    directory = "final_path"
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, 'path')
    np.save(file_path, np.array(shortest_path))

    #visualizer = Visualize_UR(ur_params, env=env, transform=transform, bb=bb) - there is already before

    try:
        path = np.load(file_path + '.npy')
        visualizer.show_path(path)
    except:
        print('No Path Found')
    
    visualizer_gif = Visualize_Gif(ur_params, env, transform, bb)#TODO - need to check if it works

