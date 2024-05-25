import numpy as np
import os
from environment import Environment
from kinematics import UR5e_PARAMS, Transform
from building_blocks import Building_Blocks
from Code.visualizer_gif import VisualizeGif
from visualizer import Visualize_UR

def main():
    ur_params = UR5e_PARAMS(inflation_factor=1)
    transform = Transform(ur_params)

    # --------- configurations-------------
    home = np.deg2rad([0, -90, 0, -90, 0, 0])

    cube1_approach = np.deg2rad([68.8, -68.3, 84.2, -107.1, -90, -18])  # approach without collision
    cube1 = np.deg2rad([69, -63, 85, -107.7, -91.3, -18.2])  # actual position of the cube

    cube2_approach = np.deg2rad([87.5, -45.5, 47.7, -102, -90.6, 3.3])  # approach without collision
    cube2 = np.deg2rad([86.9, -40.1, 47.7, -102, -90.6, 3.3])  # actual position of the cube

    cube3_approach = np.deg2rad([79.7, -46.9, 69.7, -105.1, -92.6, -10.1])  # approach without collision
    cube3 = np.deg2rad([80.2, -43.4, 68.9, -107.1, -93.9, -9.4])  # actual position of the cube

    cube4_approach = np.deg2rad([97.6, -38.3, -52.1, -100.8, -90.1, 8.5])  # approach without collision
    cube4 = np.deg2rad([97.6, -38.3, 52.1, -100.8, -90.1, 8.5])  # actual position of the cube

    cube5_approach = np.deg2rad([104.6, -85.3, 87.7, -90.5, -88.3, 17])  # approach without collision
    cube5 = np.deg2rad([105.1, -86.3, 97.4, -102, -89.8, 19.3])  # actual position of the cube

    cube6_approach = np.deg2rad([78.3, -61.7, 120.9, -87.6, -12.9, 27.7])  # approach without collision
    cube6 = np.deg2rad([78, -56.6, 120.9, -93, -12.7, 29.4])  # actual position of the cube

    cubes_approcehs = [cube1_approach, cube2_approach, cube3_approach, cube4_approach, cube5_approach, cube6_approach]
    cubes = [cube1, cube2, cube3, cube4, cube5, cube6]

    cube1_coords = [-0.10959248574268822, -0.6417732149769166, 0.1390226933317033]
    cube2_coords = [0.08539928976845282, -0.8370930220946053, 0.13813472317717034]
    cube3_coords = [-0.008445229140271685, -0.7365370847309188, 0.00955541284784159]
    cube4_coords = [0.23647185443765273, -0.769747539513382, 0.03971366463235271]
    cube5_coords = [0.26353072323141574, -0.4629969534200313, 0.2651034131371637]
    cube6_coords = [0.26940059242703984, -0.4730222745248458, 0.021688493137064376]
    initial_cubes_coords = [cube1_coords, cube2_coords, cube3_coords, cube4_coords, cube5_coords, cube6_coords]

    env = Environment(env_idx=3, cube_coords=initial_cubes_coords)

    bb = Building_Blocks(transform=transform,
                         ur_params=ur_params,
                         env=env,
                         resolution=0.1,
                         p_bias=0.05)

    visualizer = Visualize_UR(ur_params, env=env, transform=transform, bb=bb)
    gif_visualizer = VisualizeGif(ur_params, env, transform, bb)

    # Directory where your files are stored
    directory = "N_letter"

    # Load obstacle configurations
    obstacle_positions_list = np.load(os.path.join(directory, 'obstacle_positions_list.npy'), allow_pickle=True)

    # Load the plan list to understand the sequence of actions
    plan_list = eval(open(os.path.join(directory, 'plan_list.py')).read())

    obstacle_index = 0  # Index to track the current obstacle configuration
    for plan_file in plan_list:
        if plan_file.endswith('_path.npy'):
            # Load the path from the .npy file
            path = np.load(os.path.join(directory, plan_file))

            # Use the current obstacle configuration
            cubes = obstacle_positions_list[obstacle_index]

            # Generate the GIF for the current path
            gif_visualizer.save_paths_to_gif([path], [cubes], os.path.join(directory, f'{plan_file}_visualization.gif'))

            # Only increment the obstacle_index if the environment changes
            if "goal_to_cube" in plan_file:
                obstacle_index = min(obstacle_index + 1, len(obstacle_positions_list) - 1)  # Avoid going out of bounds


if __name__ == "__main__":
    main()