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

def plot_3d_path_with_tangents(path, tangents, scale=0.05):
    """Plot the path with tangents in a 3D plot."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(path[:, 0], path[:, 1], path[:, 2], label='Path')

    # Display tangents at regular intervals
    interval = len(path) // 5  # Adjust the number of tangents displayed
    indices = range(0, len(path), interval)
    for i in indices:
        x, y, z = path[i]
        dx, dy, dz = tangents[i] * scale
        ax.quiver(x, y, z, dx, dy, dz, color='r', length=scale, normalize=True)  # Normalize the vector for uniform display

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

# Main execution block
waypoints = create_sine_wave_waypoints()
splines = fit_spline(waypoints)
smooth_path = evaluate_spline(splines)
tangents = compute_tangents(splines)

plot_3d_path_with_tangents(smooth_path, tangents, scale=0.1)  # Adjust scale as needed
