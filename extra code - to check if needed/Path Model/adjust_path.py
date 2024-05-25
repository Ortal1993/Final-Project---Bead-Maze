import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline

def create_sine_wave_waypoints(num_points=100, amplitude=5, frequency=1, length=10):
    """Generate waypoints along a sine wave in 3D."""
    x = np.linspace(0, length, num_points)
    y = amplitude * np.sin(frequency * x)
    z = np.zeros_like(x)  # Keep z constant for simplicity
    return np.column_stack((x, y, z))

def fit_spline(waypoints):
    """Fit cubic splines to 3D waypoints."""
    t = np.linspace(0, 1, len(waypoints))
    splines = [CubicSpline(t, waypoints[:, dim]) for dim in range(3)]
    return splines

def evaluate_spline(splines, num_points=100):
    """Evaluate cubic splines to generate smooth path."""
    t_new = np.linspace(0, 1, num_points)
    return np.array([spline(t_new) for spline in splines]).T

def compute_tangents(splines, num_points=100):
    """Compute the tangents (derivatives) of the spline at regular intervals."""
    t_new = np.linspace(0, 1, num_points)
    derivatives = [spline(t_new, 1) for spline in splines]  # Compute the first derivative
    tangents = np.array(derivatives).T  # Transpose to align with the evaluation points
    return tangents

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

def plot_3d_path_with_tangents(path, tangents, scale=0.1, title='Path with Tangents'):
    """Plot the path with tangents in a 3D plot."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(path[:, 0], path[:, 1], path[:, 2], label='Path')

    # Display tangents
    for i in range(len(path)):
        x, y, z = path[i]
        dx, dy, dz = tangents[i] * scale
        ax.quiver(x, y, z, dx, dy, dz, color='r', length=scale, normalize=False)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title(title)
    plt.show()

def main():
    # Create sine wave waypoints
    waypoints = create_sine_wave_waypoints()

    # Fit spline to waypoints and evaluate for a smooth path
    splines = fit_spline(waypoints)
    smooth_path = evaluate_spline(splines)
    tangents = compute_tangents(splines)

    # Define new start and end points for the sine wave
    new_start = np.array([10, -10, 0])
    new_end = np.array([20, 0, 0])

    # Adjust waypoints and re-fit/re-evaluate the spline
    adjusted_waypoints = adjust_path(waypoints, new_start, new_end)
    adjusted_splines = fit_spline(adjusted_waypoints)
    adjusted_smooth_path = evaluate_spline(adjusted_splines)
    adjusted_tangents = compute_tangents(adjusted_splines)

    # Plot both paths with tangents
    plot_3d_path_with_tangents(smooth_path, tangents, scale=0.05, title='Original Path with Tangents')
    plot_3d_path_with_tangents(adjusted_smooth_path, adjusted_tangents, scale=0.05, title='Adjusted Path with Tangents')

if __name__ == "__main__":
    main()
