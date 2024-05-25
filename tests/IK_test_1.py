import numpy as np

# Assumed imports (make sure these classes and methods are correctly defined in your project)
from kinematics import UR5e_PARAMS, Transform
from inverse_kinematics import Inverse_Kinematics
from environment import Environment  # This should include your collision detection and environment setup

def main():
    # Define the target position and orientation
    tx, ty, tz = 0.3, -0.2, 0.5  # Target position
    tangent = np.array([1, 1, 2])  # Example tangent vector
    # alpha, beta, gamma = calculate_direction_angles(tangent)  # Calculate direction angles
    # print(np.rad2deg(alpha), np.rad2deg(beta), np.rad2deg(gamma))
    # Initialize the IK solver with the target pose
    ik_solver = Inverse_Kinematics(tx, ty, tz, tangent)

    # Compute the IK solutions
    possible_configs = ik_solver.find_possible_configs()

    # Check results using Forward Kinematics
    if len(possible_configs) > 0:
        print("IK Solutions found:")
        for config in possible_configs:
            fk_result = ik_solver.forward_kinematic_solution(ik_solver.DH_matrix_UR5e, config)
            print(f"Configuration: {config}")
            print(f"FK Result: {fk_result}")

            # Check if the FK result matches the desired pose
            end_effector_pos = np.array([fk_result[0, 3], fk_result[1, 3], fk_result[2, 3]])
            target_pos = np.array([tx, ty, tz])
            if np.allclose(end_effector_pos, target_pos, atol=0.01):
                print("Success: FK result matches the target position.")
            else:
                print("Error: FK result does not match the target position.")
        print("No valid IK solutions found.")

if __name__ == "__main__":
    main()
