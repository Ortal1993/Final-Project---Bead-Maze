import numpy as np

def normalize_vector(vector):
    """Normalize the vector to unit length."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Cannot normalize the zero vector")
    return vector / norm

def calculate_direction_angles(vector):
    """Calculate direction angles from a 3D vector."""
    normalized_vector = normalize_vector(vector)
    alpha = np.arccos(normalized_vector[0])  # Angle with x-axis
    beta = np.arccos(normalized_vector[1])  # Angle with y-axis
    gamma = np.arccos(normalized_vector[2])  # Angle with z-axis
    return np.degrees(alpha), np.degrees(beta), np.degrees(gamma)


def calculate_normal_vector(tangent):
    """Calculate a vector normal to the provided tangent vector."""
    # Ensure the tangent is normalized
    tangent = normalize_vector(tangent)
    # Choose an arbitrary vector that is not aligned with the tangent
    arbitrary_vector = np.array([0, 0, 1]) if not np.allclose(tangent, [0, 0, 1]) else np.array([1, 0, 0])
    # Calculate the normal vector using the cross product
    normal_vector = np.cross(tangent, arbitrary_vector)
    # Normalize the normal vector
    normal_vector /= np.linalg.norm(normal_vector)
    return normal_vector
# Example usage   [ 0.70710678 -0.70710678  0.        ]
tangent = np.array([1, 2, 0])  # Example tangent at a path point
normal_vector =[ 0.70710678, -0.70710678,  0.        ]
alpha, beta, gamma = calculate_direction_angles(normal_vector)
print("Normal Vector:", normal_vector)
print("Direction Angles (Degrees): Alpha =", alpha, "Beta =", beta, "Gamma =", gamma)
print("Direction Angles (rad): Alpha =", np.deg2rad(alpha), "Beta =", np.deg2rad(beta), "Gamma =", np.deg2rad(gamma))
