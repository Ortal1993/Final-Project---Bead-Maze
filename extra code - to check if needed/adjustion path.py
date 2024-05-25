import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline


def fit_spline(waypoints):
    """Fit cubic splines to 3D waypoints."""
    t = np.linspace(0, 1, len(waypoints))
    splines = [CubicSpline(t, waypoints[:, dim]) for dim in range(3)]
    return splines


def evaluate_spline(splines, num_points=100):
    """Evaluate cubic splines to generate smooth path."""
    t_new = np.linspace(0, 1, num_points)
    return np.array([spline(t_new) for spline in splines]).T


def rotate_about_z(points, theta):
    """Rotate points around the Z-axis by theta degrees."""
    theta = np.deg2rad(theta)  # Convert to radians
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    return np.dot(points, rotation_matrix.T)


def translate(points, offset):
    """Translate points by a given offset."""
    return points + offset


def adjust_path(waypoints, new_start, new_end):
    """Adjust the path to a new start and end position."""
    orig_start = waypoints[0]
    orig_end = waypoints[-1]

    translation_vector = new_start - orig_start
    translated_points = translate(waypoints, translation_vector)

    orig_vector = orig_end - orig_start
    new_vector = new_end - new_start
    orig_angle = np.arctan2(orig_vector[1], orig_vector[0])
    new_angle = np.arctan2(new_vector[1], new_vector[0])
    rotation_angle = np.rad2deg(new_angle - orig_angle)

    rotated_translated_points = rotate_about_z(translated_points, rotation_angle)
    return rotated_translated_points


def plot_3d_path(paths, labels=['Original Path', 'Adjusted Path']):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for path, label in zip(paths, labels):
        ax.plot(path[:, 0], path[:, 1], path[:, 2], label=label)
        ax.scatter(path[0, 0], path[0, 1], path[0, 2], color='red')  # Start point
        ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], color='green')  # End point
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


# Example usage
waypoints = np.array([
    [0, 0, 0],
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
new_start = np.array([10, 10, 0])
new_end = np.array([17, 18, 9])

splines = fit_spline(waypoints)
smooth_path = evaluate_spline(splines)
adjusted_waypoints = adjust_path(waypoints, new_start, new_end)
adjusted_splines = fit_spline(adjusted_waypoints)
adjusted_smooth_path = evaluate_spline(adjusted_splines)

plot_3d_path([smooth_path, adjusted_smooth_path])
