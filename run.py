import numpy as np
import argparse
from inverse_kinematics import Inverse_Kinematics
from layered_graph import Layered_Graph
from path_model import Path_Model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='script for bead maze')
    parser.add_argument('-waypoints', '--waypoints', type=str, default='way_points.json',
                        help='Json file name containing all way points')
    args = parser.parse_args()

    path = Path_Model(json_file=args.waypoints)
    splines, smooth_path, tangents = path.process_path()
    waypoints_coords = path.get_waypoints_coords()

    # Initialize the inverse kinematics solver with initial parameters
    tx, ty, tz = waypoints_coords[0]
    ik_solver = Inverse_Kinematics(tx, ty, tz, tangents[0])

    ik_solutions_per_layer = []
    # Process each waypoint and its corresponding tangent
    for waypoint, tangent in zip(smooth_path, tangents):
        tx, ty, tz = waypoint
        # Update IK solver target and tangent
        ik_solver.update_target_and_tangent(tx, ty, tz, tangent)

        # Compute IK solutions for the current waypoint
        possible_configs = ik_solver.find_possible_configs()

        ik_solutions_per_layer.append(possible_configs)

    # Building the graph using valid configurations for each waypoint
    graph = Layered_Graph(ik_solver, splines)
    graph.build_graph(ik_solutions_per_layer)


    # TODO create Dijkstra algorithm to find an optimal path
