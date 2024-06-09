import numpy as np
from scipy.spatial.distance import cdist
from visualizer import Visualize_UR

X = 0
Y = 1
Z = 2
MIN_RESOLUTION = 3


class Building_Blocks(object):
    def __init__(self, transform, ur_params, env, inverse_kinematics, resolution=0.1, p_bias=0.05):
        self.transform = transform
        self.ur_params = ur_params
        self.env = env
        self.inverse_kinematics = inverse_kinematics
        self.resolution = resolution
        self.p_bias = p_bias
        self.cost_weights = np.array([0.4, 0.3, 0.2, 0.1, 0.07, 0.05])

        self.checked_confs = {}  # dict[conf] = bool #dict[conf] = bool, true if there is a collision and false otherwise

        self.possible_joints_collision = []
        for i in range(len(self.ur_params.ur_links)):
            for j in range(i + 2, len(self.ur_params.ur_links)):
                if self.ur_params.ur_links[i] == 'wrist_1_link':
                    break
                self.possible_joints_collision.append((self.ur_params.ur_links[i], self.ur_params.ur_links[j]))

    def is_in_collision(self, conf) -> bool:
        """
        check for collision in given configuration, arm-arm and arm-obstacle
        return True if in collision
        @param conf - some configuration
        """
        # Ensure conf is a 1D array
        conf = np.asarray(conf).reshape(-1)

        tupl_conf = tuple(conf)  # Convert the configuration to a tuple
        try:
            res = self.checked_confs[tupl_conf]
            return res
        except KeyError:
            self.checked_confs[tupl_conf] = True

        global_sphere_coords = self.transform.conf2sphere_coords(conf)

        sphere_radius_per_joint = self.ur_params.sphere_radius

        # arm - floor/window collision
        for joint, spheres in global_sphere_coords.items():
            if joint == 'shoulder_link':
                continue
            if any(((sphere[2] - sphere_radius_per_joint[joint]) <= 0) or (
                    sphere[0] + sphere_radius_per_joint[joint] > 0.4) for sphere in spheres):
                return True

        # arm - arm collision
        spheres_per_joint = {joint: np.array(global_sphere_coords[joint]) for joint in self.ur_params.ur_links}
        for joint_1, joint_2 in self.possible_joints_collision:
            robot_spheres_1 = np.array([np.array(spheres[:3], dtype=float) for spheres in spheres_per_joint[joint_1]])
            robot_spheres_2 = np.array([np.array(spheres[:3], dtype=float) for spheres in spheres_per_joint[joint_2]])
            distances = cdist(robot_spheres_1, robot_spheres_2)
            sum_of_radii = sphere_radius_per_joint[joint_1] + sphere_radius_per_joint[joint_2]
            if np.any(distances <= sum_of_radii):
                return True

        # arm - obstacle collision
        obstacles = self.env.obstacles
        if obstacles.size > 0:
            for joint in self.ur_params.ur_links:
                robot_spheres = np.array([np.array(spheres[:3], dtype=float) for spheres in spheres_per_joint[joint]])
                distances = cdist(robot_spheres, obstacles)
                sum_of_radii = sphere_radius_per_joint[joint] + self.env.radius
                if np.any(distances <= sum_of_radii):
                    return True

        self.checked_confs[tupl_conf] = False
        return False

    def local_planner(self, prev_conf, current_conf, splines):
        """Check for collisions and path deviation between two configurations."""
        dist_prev_curr = np.linalg.norm(current_conf - prev_conf)
        num_intermediate_configs = max(int(np.ceil(dist_prev_curr / self.resolution)), MIN_RESOLUTION)
        intermediate_configs = np.linspace(prev_conf, current_conf, num_intermediate_configs)

        max_dist = 0

        for config in intermediate_configs:
            config = np.asarray(config).reshape(-1)  # Ensure config is a 1D array
            if self.is_in_collision(config):
                return True, np.inf

            # Compute end effector position using forward kinematics from Inverse_Kinematics
            transformation_matrix = self.inverse_kinematics.forward_kinematic_solution(
                self.inverse_kinematics.DH_matrix_UR5e, config)
            end_effector_pos = transformation_matrix[:3, 3].flatten()  # Ensure end_effector_pos is a 1D array

            # Calculate distance from the path
            current_dist = self.compute_closest_point_on_path(end_effector_pos, splines)
            if current_dist > 10:
                return True, np.inf
            max_dist = max(max_dist, current_dist)

        return False, max_dist

    def compute_closest_point_on_path(self, point, splines):
        """
        Compute the closest point on the path (defined by splines) to the given point.
        This function should return the maximum distance from the point to the spline path.
        """
        point = point.flatten()  # Ensure point is a 1D array
        distances = np.linalg.norm(splines - point, axis=1)
        return np.min(distances)

    def edge_cost(self, conf1, conf2):
        """Returns the Edge cost- the cost of transition from configuration 1 to configuration 2."""
        return np.linalg.norm(conf1 - conf2)
