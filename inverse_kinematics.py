from environment import Environment
from kinematics import UR5e_PARAMS, Transform
from building_blocks import Building_Blocks
import numpy as np
from math import pi, cos, sin, atan2, acos, sqrt, asin

class Inverse_Kinematics(object):
    def __init__(self, tx, ty, tz, tangent):
        self.tool_length = 0.135  # meters, adjust if different
        self.DH_matrix_UR5e = np.matrix([
            [0, pi / 2.0, 0.1625],
            [-0.425, 0, 0],
            [-0.3922, 0, 0],
            [0, pi / 2.0, 0.1333],
            [0, -pi / 2.0, 0.0997],
            [0, 0, 0.0996 + self.tool_length]
        ])
        self.env_idx = 3
        self.update_target_and_tangent(tx, ty, tz, tangent)

    def update_target_and_tangent(self, tx, ty, tz, tangent):
        self.tx = tx
        self.ty = ty
        self.tz = tz
        self.tangent = tangent
        # Compute the normal vector from the tangent
        normal_vector = self.calculate_normal_vector(tangent)
        # Calculate direction angles from the normal vector
        self.alpha, self.beta, self.gamma = self.calculate_direction_angles(normal_vector)

    def normalize_vector(self, vector):
        """Normalize the vector to unit length."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Cannot normalize the zero vector")
        return vector / norm

    def calculate_direction_angles(self, vector):
        """Calculate direction angles from a 3D vector."""
        normalized_vector = self.normalize_vector(vector)
        alpha = acos(normalized_vector[0])  # Angle with x-axis
        beta = acos(normalized_vector[1])  # Angle with y-axis
        gamma = acos(normalized_vector[2])  # Angle with z-axis
        return alpha, beta, gamma

    def calculate_normal_vector(self, tangent):
        """Calculate a vector normal to the provided tangent vector."""
        # Ensure the tangent is normalized
        normalized_tangent = self.normalize_vector(tangent)
        arbitrary_vector = np.array([0, 0, 1]) if not np.allclose(normalized_tangent, [0, 0, 1]) else np.array([1, 0, 0])
        normal_vector = np.cross(normalized_tangent, arbitrary_vector)
        return self.normalize_vector(normal_vector)

    def compute_transformation_matrix(self):
        """Compute the transformation matrix based on position and Euler angles."""
        R_x = np.matrix([
            [1, 0, 0],
            [0, cos(self.alpha), -sin(self.alpha)],
            [0, sin(self.alpha), cos(self.alpha)]
        ])
        R_y = np.matrix([
            [cos(self.beta), 0, sin(self.beta)],
            [0, 1, 0],
            [-sin(self.beta), 0, cos(self.beta)]
        ])
        R_z = np.matrix([
            [cos(self.gamma), -sin(self.gamma), 0],
            [sin(self.gamma), cos(self.gamma), 0],
            [0, 0, 1]
        ])
        R = R_z * R_y * R_x
        T = np.matrix([[self.tx], [self.ty], [self.tz]])

        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = R
        transformation_matrix[:3, 3] = T.flatten()

        return transformation_matrix

    def mat_transform_DH(self, DH_matrix, n, edges=np.matrix([[0], [0], [0], [0], [0], [0]])):
        n = n - 1
        t_z_theta = np.matrix([[cos(edges[n]), -sin(edges[n]), 0, 0],
                               [sin(edges[n]), cos(edges[n]), 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], copy=False)
        t_zd = np.matrix(np.identity(4), copy=False)
        t_zd[2, 3] = DH_matrix[n, 2]
        t_xa = np.matrix(np.identity(4), copy=False)
        t_xa[0, 3] = DH_matrix[n, 0]
        t_x_alpha = np.matrix([[1, 0, 0, 0],
                               [0, cos(DH_matrix[n, 1]), -sin(DH_matrix[n, 1]), 0],
                               [0, sin(DH_matrix[n, 1]), cos(DH_matrix[n, 1]), 0],
                               [0, 0, 0, 1]], copy=False)
        transform = t_z_theta * t_zd * t_xa * t_x_alpha
        return transform

    def forward_kinematic_solution(self, DH_matrix, edges=np.matrix([[0], [0], [0], [0], [0], [0]])):
        t01 = self.mat_transform_DH(DH_matrix, 1, edges)
        t12 = self.mat_transform_DH(DH_matrix, 2, edges)
        t23 = self.mat_transform_DH(DH_matrix, 3, edges)
        t34 = self.mat_transform_DH(DH_matrix, 4, edges)
        t45 = self.mat_transform_DH(DH_matrix, 5, edges)
        t56 = self.mat_transform_DH(DH_matrix, 6, edges)
        answer = t01 * t12 * t23 * t34 * t45 * t56
        return answer

    def inverse_kinematic_solution(self, DH_matrix, transform_matrix):
        theta = np.matrix(np.zeros((6, 8)))
        # theta 1
        T06 = transform_matrix

        P05 = T06 * np.matrix([[0], [0], [-DH_matrix[5, 2]], [1]])
        psi = atan2(P05[1], P05[0])
        phi_cos = (DH_matrix[1, 2] + DH_matrix[3, 2] + DH_matrix[2, 2]) / sqrt(P05[0] ** 2 + P05[1] ** 2)
        phi = acos(np.clip(phi_cos, -1.0, 1.0))  # Use np.clip to avoid domain error
        theta[0, 0:4] = psi + phi + pi / 2
        theta[0, 4:8] = psi - phi + pi / 2

        # theta 5
        for i in {0, 4}:
            th5cos = (T06[0, 3] * sin(theta[0, i]) - T06[1, 3] * cos(theta[0, i]) - (
                        DH_matrix[1, 2] + DH_matrix[3, 2] + DH_matrix[2, 2])) / DH_matrix[5, 2]
            if 1 >= th5cos >= -1:
                th5 = acos(th5cos)
            else:
                th5 = 0
            theta[4, i:i + 2] = th5
            theta[4, i + 2:i + 4] = -th5
        # theta 6
        for i in {0, 2, 4, 6}:
            T60 = np.linalg.inv(T06)
            th = atan2((-T60[1, 0] * sin(theta[0, i]) + T60[1, 1] * cos(theta[0, i])),
                       (T60[0, 0] * sin(theta[0, i]) - T60[0, 1] * cos(theta[0, i])))
            theta[5, i:i + 2] = th

        # theta 3
        for i in {0, 2, 4, 6}:
            T01 = self.mat_transform_DH(DH_matrix, 1, theta[:, i])
            T45 = self.mat_transform_DH(DH_matrix, 5, theta[:, i])
            T56 = self.mat_transform_DH(DH_matrix, 6, theta[:, i])
            T14 = np.linalg.inv(T01) * T06 * np.linalg.inv(T45 * T56)
            P13 = T14 * np.matrix([[0], [-DH_matrix[3, 2]], [0], [1]])
            costh3 = ((P13[0] ** 2 + P13[1] ** 2 - DH_matrix[1, 0] ** 2 - DH_matrix[2, 0] ** 2) /
                      (2 * DH_matrix[1, 0] * DH_matrix[2, 0]))
            if 1 >= costh3 >= -1:
                th3 = acos(costh3)
            else:
                th3 = 0
            theta[2, i] = th3
            theta[2, i + 1] = -th3

        # theta 2,4
        for i in range(8):
            T01 = self.mat_transform_DH(DH_matrix, 1, theta[:, i])
            T45 = self.mat_transform_DH(DH_matrix, 5, theta[:, i])
            T56 = self.mat_transform_DH(DH_matrix, 6, theta[:, i])
            T14 = np.linalg.inv(T01) * T06 * np.linalg.inv(T45 * T56)
            P13 = T14 * np.matrix([[0], [-DH_matrix[3, 2]], [0], [1]])

            theta[1, i] = atan2(-P13[1], -P13[0]) - asin(
                -DH_matrix[2, 0] * sin(theta[2, i]) / sqrt(P13[0] ** 2 + P13[1] ** 2)
            )
            T32 = np.linalg.inv(self.mat_transform_DH(DH_matrix, 3, theta[:, i]))
            T21 = np.linalg.inv(self.mat_transform_DH(DH_matrix, 2, theta[:, i]))
            T34 = T32 * T21 * T14
            theta[3, i] = atan2(T34[1, 0], T34[0, 0])
        return theta

    def vector_to_euler_angles(self, tangent):
        norm = np.linalg.norm(tangent)
        if norm == 0:
            return 0, 0, 0
        tangent /= norm
        alpha = 0  # Roll is often irrelevant for simple path following
        beta = np.arctan2(tangent[2], np.sqrt(tangent[0] ** 2 + tangent[1] ** 2))
        gamma = np.arctan2(tangent[1], tangent[0])
        return alpha, beta, gamma

    def find_possible_configs(self):
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        env_idx = self.env_idx
        tx = self.tx
        ty = self.ty
        tz = self.tz

        transform = np.matrix([[cos(beta) * cos(gamma), sin(alpha) * sin(beta)*cos(gamma) - cos(alpha)*sin(gamma),
                                cos(alpha)*sin(beta)*cos(gamma)+sin(alpha)*sin(gamma), tx],
                               [cos(beta)* sin(gamma), sin(alpha)*sin(beta)*sin(gamma)+cos(alpha)*cos(gamma),
                                cos(alpha)*sin(beta)*sin(gamma)-sin(alpha)*cos(gamma), ty],
                               [-sin(beta), sin(alpha)*cos(beta), cos(alpha)*cos(beta), tz],
                               [0, 0,0,1]])

        IKS = self.inverse_kinematic_solution(self.DH_matrix_UR5e, transform)

        ur_params = UR5e_PARAMS(inflation_factor=1)
        env = Environment(env_idx=env_idx)
        transform = Transform(ur_params)
        bb = Building_Blocks(transform=transform, ur_params=ur_params, env=env, inverse_kinematics=self, resolution=0.1, p_bias=0.05)

        candidate_sols = []
        for i in range(IKS.shape[1]):
            candidate_sols.append(IKS[:, i])
        candidate_sols = np.array(candidate_sols)

        # check for collisions and angles limits
        sols = []
        for candidate_sol in candidate_sols:
            candidate_sol_array = candidate_sol.flatten()
            if bb.is_in_collision(candidate_sol_array):
                continue
            for idx, angle in enumerate(candidate_sol):
                if 2*np.pi > angle > np.pi:
                    candidate_sol[idx] = -(2*np.pi - angle)
                if -2*np.pi < angle < -np.pi:
                    candidate_sol[idx] = -(2*np.pi + angle)
            if np.max(candidate_sol) > np.pi or np.min(candidate_sol) < -np.pi:
                continue
            sols.append(candidate_sol)

        # verify solution:
        final_sol = []
        for sol in sols:
            transform = self.forward_kinematic_solution(self.DH_matrix_UR5e, sol)
            diff = np.linalg.norm(np.array([transform[0,3],transform[1,3],transform[2,3]])-np.array([tx,ty,tz]))
            if diff < 0.05:
                final_sol.append(sol)
        final_sol = np.array(final_sol)
        return final_sol
