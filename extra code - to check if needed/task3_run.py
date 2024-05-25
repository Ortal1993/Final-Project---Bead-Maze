import numpy as np
from environment import Environment
from kinematics import UR5e_PARAMS, Transform
from planners import RRT_STAR
from building_blocks import Building_Blocks
from visualizer import Visualize_UR
from Code.visualizer_gif import VisualizeGif
from inverse_kinematics import Inverse_Kinematics
import math
import os

def main():
    ur_params = UR5e_PARAMS(inflation_factor=1)
    transform = Transform(ur_params)

    # --------- configurations-------------
    home = np.deg2rad([0, -90, 0, -90, 0,0 ])

    cube1_approach = np.deg2rad([68.8,-68.3, 84.2,-107.1,-90,-18]) #approach without collision
    cube1 = np.deg2rad([69,-63,85, -107.7,-91.3,-18.2]) #actual position of the cube 
    
    cube2_approach = np.deg2rad([87.5, -45.5, 47.7, -102, -90.6, 3.3]) #approach without collision
    cube2 = np.deg2rad([86.9, -40.1, 47.7, -102, -90.6, 3.3]) #actual position of the cube 

    cube3_approach = np.deg2rad([79.7,-46.9,69.7,-105.1,-92.6,-10.1]) #approach without collision
    cube3 = np.deg2rad([80.2, -43.4, 68.9, -107.1,-93.9, -9.4]) #actual position of the cube 
 
    cube4_approach = np.deg2rad([97.6, -38.3,-52.1, -100.8, -90.1, 8.5]) #approach without collision
    cube4 = np.deg2rad([97.6, -38.3, 52.1, -100.8, -90.1, 8.5]) #actual position of the cube
 
    cube5_approach = np.deg2rad([104.6, -85.3,87.7, -90.5,-88.3, 17]) #approach without collision
    cube5 = np.deg2rad([105.1, -86.3, 97.4, -102, -89.8, 19.3]) #actual position of the cube 

    cube6_approach = np.deg2rad([78.3, -61.7, 120.9, -87.6,-12.9, 27.7]) #approach without collision
    cube6 = np.deg2rad([78,-56.6,120.9, -93,-12.7,29.4]) #actual position of the cube

    cubes_approcehs = [cube1_approach, cube2_approach, cube3_approach, cube4_approach, cube5_approach, cube6_approach]
    cubes = [cube1, cube2, cube3, cube4, cube5, cube6]  
    
    cube1_coords = [-0.10959248574268822, -0.6417732149769166, 0.1390226933317033]
    cube2_coords = [0.08539928976845282, -0.8370930220946053, 0.13813472317717034]
    cube3_coords =  [-0.008445229140271685, -0.7365370847309188, 0.00955541284784159]
    cube4_coords = [0.23647185443765273 ,-0.769747539513382, 0.03971366463235271]
    cube5_coords =[0.26353072323141574 ,-0.4629969534200313, 0.2651034131371637]
    cube6_coords =  [0.26940059242703984, -0.4730222745248458, 0.021688493137064376]
    cube7_coords = [-0.28, -0.45, 0.1]

    initial_cubes_coords = [cube1_coords,cube2_coords,cube3_coords,cube4_coords,cube5_coords,cube6_coords]


    env = Environment(env_idx=3, cube_coords=initial_cubes_coords)
    # env = Environment(env_idx=3, cube_coords=cubes_test)
    
    bb = Building_Blocks(transform=transform, 
                        ur_params=ur_params, 
                        env=env,
                        resolution=0.1, 
                        p_bias=0.05)
    
    visualizer = Visualize_UR(ur_params, env=env, transform=transform, bb=bb)
    gif_visualizer = VisualizeGif(ur_params, env, transform, bb)


    #for O letter
    letter_locations = [(-0.145, -0.24), (-0.29, -0.33), (-0.29, -0.42), (-0.145, -0.51), (0.0, -0.42), (0.0, -0.33)]
    files_names = ['1_home_to_cube1', '2_cube1_to_cube1goal', '3_cube1goal_to_cube2', '4_cube2_to_cube2goal', '5_cube2goal_to_cube3',
                   '6_cube3_to_cube3goal', '7_cube3goal_to_cube4', '8_cube4_to_cube4goal', '9_cube4goal_to_cube5', '10_cube5_to_cube5goal',
                   '11_cube5goal_to_cube6', '12_cube6_to_cube6goal']
    # For N letter, we will assume that (-0.28, -0.45) is already there(cube7_coords)
    n_letter_locations = [ (-0.28, -0.35), (-0.28, -0.25), (-0.15, -0.35), (-0.02, -0.25), (-0.02, -0.35), (-0.02, -0.45)]
    # Adding obstacle to create N letter
    # initial_cubes_coords.append(cube7_coords)

    obstacle_positions_list = []
    plan_list = []
    plan_list.append('open')

    curr_location = home


    for i, cube in enumerate(cubes):
        
        path_home_to_cube1 = go_to_cube(curr_location, cubes_approcehs[i], None, cube, files_names[2 * i], bb, visualizer)
        plan_list.append(files_names[2 * i] + '_path.npy')
        plan_list.append('close')
        # for O letter
        cube_goal = get_cube_goal(letter_locations[i], cubes_approcehs[i], bb)
        # For N letter
        # cube_goal = get_cube_goal(n_letter_locations[i], cubes_approcehs[i], bb)

        #go to cube1_goal
        path_cube1_to_cube1goal = go_to_cube(cubes_approcehs[i], cube_goal, cube, None, files_names[(2 * i) + 1], bb, visualizer)
        plan_list.append(files_names[(2 * i) + 1] + '_path.npy')
        plan_list.append('open')
        obstacle_positions_list.append(list(initial_cubes_coords))
        #updating the envirnment
        initial_cubes_coords[i] = get_cube_coords(cube_goal, transform)
        visualizer.env = Environment(env_idx=3, cube_coords=initial_cubes_coords)
        bb.env = Environment(env_idx=3, cube_coords=initial_cubes_coords)

        curr_location = cube_goal



    #TODO - ortal - need to check add_before / add_after?
    obstacle_positions_list.append(list(initial_cubes_coords))
    # go to home
    path_cube6goal_to_home = go_to_cube(curr_location, home, None, None, '13_cube6goal_to_home', bb, visualizer)
    plan_list.append('13_cube6goal_to_home_path.npy')
    plan_list.append('close')#TODO - ortal - check if need to be closed at the and


    directory = "O_letter"
    # directory = "N_letter"

    np.save(os.path.join(directory, 'obstacle_positions_list.npy'), obstacle_positions_list)
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    plan_list_filename = 'plan_list.py'
    file_path = os.path.join(directory, plan_list_filename)
    with open(file_path, 'w') as f:
        #f.writelines(plan_list)  # Convert list to its string representation
        f.write(repr(plan_list))
        f.close()
        # Directory where your files are stored

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
            gif_visualizer.save_paths_to_gif([path], [cubes],
                                             os.path.join(directory, f'{plan_file}_visualization.gif'))

            # Only increment the obstacle_index if the environment changes
            if "goal_to_cube" in plan_file:
                obstacle_index = min(obstacle_index + 1,
                                     len(obstacle_positions_list) - 1)  # Avoid going out of bounds


def go_to_cube(start_conf, goal_conf, add_before_conf, add_after_conf, filename, bb, visualizer):
    rrt_star_planner = RRT_STAR(max_step_size=0.5,
                                max_itr=800,
                                bb=bb)#TODO - change back to 10000
    
    rrt_start = start_conf
    rrt_goal = goal_conf
    add_before = add_before_conf #add before path
    add_after = add_after_conf #add after path
    filename = filename

    if bb.is_in_collision(rrt_start):
        print('start in collision')
    if bb.is_in_collision(rrt_goal): 
        print('goal in collision')
    
    rrt_path, _ = rrt_star_planner.find_path(start_conf=rrt_start,
                                        goal_conf=rrt_goal)
    
    # directory = "O_letter" #TODO change to "N_letter"

    directory = "O_letter"
    # directory = "N_letter"

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    path = []
    if add_before is not None:
        path.append(add_before)
    for conf in rrt_path:
        path.append(conf)
    if add_after is not None:
        path.append(add_after)
    file_path = os.path.join(directory, filename +'_path')
    np.save(file_path, np.array(path))             

    try:
        path = np.load(file_path + '.npy')
        print("it is work")
        # visualizer.show_path(path)
        #visualizer.show_conf(path[-1])
    except:
        print('No Path Found')

    formatted_rows = ['[' + ', '.join(map(str, row)) + ']' for row in path]
    formatted_path = ',\n'.join(formatted_rows)
    print("path " + filename + ": ")
    print('[' + formatted_path + ']')
    print("\n")
    return path

def get_cube_goal(xy, cube_approach, bb):
    in_kinematics = Inverse_Kinematics(xy[0], xy[1])
    cube_goals = in_kinematics.find_possible_configs()
    cube_goal = None
    min_cost = math.inf
    for goal in cube_goals:
        rrt_star_planner = RRT_STAR(max_step_size=0.5,
                                max_itr=800,
                                bb=bb)#TODO - change back to 10000
        array_goal = goal.reshape(6)
        _, cost = rrt_star_planner.find_path(start_conf=cube_approach,
                                        goal_conf=array_goal)
        if cost < min_cost:
            min_cost = cost
            cube_goal = array_goal
    
    cube_goal = cube_goals[0].reshape(6)
    """print("cube_goal: ", cube_goal)"""
    return cube_goal

def get_cube_coords(cube_goal, transform):
    manipulator_spheres = transform.conf2sphere_coords(cube_goal) # forward kinematics
    cube_coords_temp = manipulator_spheres['wrist_3_link'][-1][:3]
    cube_coords = cube_coords_temp.tolist()
    return cube_coords

if __name__ == '__main__':
    main()



