import numpy as np

def normalize_vector(vector):
    """Normalize the vector to have unit length."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Cannot normalize the zero vector")
    return vector / norm

def calculate_angles_with_axes(vector):
    """Calculate the angles between the vector and each of the x, y, and z axes."""
    unit_vector = normalize_vector(vector)
    return np.arccos(unit_vector)  # Return radians directly

def rotation_matrix_x(angle):
    """Return the rotation matrix for a rotation around the x-axis by 'angle' radians."""
    return np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])

def rotation_matrix_y(angle):
    """Return the rotation matrix for a rotation around the y-axis by 'angle' radians."""
    return np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])

def rotation_matrix_z(angle):
    """Return the rotation matrix for a rotation around the z-axis by 'angle' radians."""
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])

def is_orthonormal(matrix):
    """Check if a matrix is orthonormal."""
    identity = np.eye(3)
    return np.allclose(identity, matrix @ matrix.T) and np.allclose(identity, matrix.T @ matrix)

# Example usage

tangent = np.array([1, -1, np.sqrt(2)])
angles = calculate_angles_with_axes(tangent)
print("sum of cos", np.cos(angles[0])**2+np.cos(angles[1])**2+np.cos(angles[2])**2)
print("angles",  np.rad2deg(angles))

# Get rotation matrices
Rx = rotation_matrix_x(angles[0])
Ry = rotation_matrix_y(angles[1])
Rz = rotation_matrix_z(angles[2])

# Combine rotation matrices
combined_rotation_matrix = Rz @ Ry @ Rx

# Print the resulting matrix
print("Combined Rotation Matrix:")
print(combined_rotation_matrix)

# Check if the resulting matrix is orthonormal
orthonormal = is_orthonormal(combined_rotation_matrix)
print("Is the combined matrix orthonormal?", orthonormal)
