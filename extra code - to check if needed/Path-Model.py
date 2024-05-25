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

class PathModel:
    def __init__(self, waypoints, new_start, new_end):
        self.waypoints = waypoints
        self.new_start = np.array(new_start)
        self.new_end = np.array(new_end)

    def fit_spline(self, waypoints):
        """Fit cubic splines to provided 3D waypoints."""
        t = np.linspace(0, 1, len(waypoints))
        return [CubicSpline(t, waypoints[:, dim]) for dim in range(3)]

    def evaluate_spline(self, splines, num_points=100):
        """Evaluate cubic splines to generate smooth path."""
        t_new = np.linspace(0, 1, num_points)
        return np.array([spline(t_new) for spline in splines]).T

    def compute_tangents(self, splines, num_points=100):
        """Compute the tangents (derivatives) of the spline at regular intervals."""
        t_new = np.linspace(0, 1, num_points)
        derivatives = [spline(t_new, 1) for spline in splines]
        return np.array(derivatives).T

    def rotate_about_z(self, points, theta):
        """Rotate points around the Z-axis by theta degrees."""
        theta = np.deg2rad(theta)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        return np.dot(points, rotation_matrix.T)

    def translate(self, points, offset):
        """Translate points by a given offset."""
        return points + offset

    def adjust_path(self):
        """Adjust the path to align with new start and end points."""
        translated_points = self.translate(self.waypoints, self.new_start - self.waypoints[0])
        orig_vector = self.waypoints[-1] - self.waypoints[0]
        new_vector = self.new_end - self.new_start
        orig_angle = np.arctan2(orig_vector[1], orig_vector[0])
        new_angle = np.arctan2(new_vector[1], new_vector[0])
        rotation_angle = new_angle - orig_angle
        return self.rotate_about_z(translated_points, np.rad2deg(rotation_angle))

    def plot_3d_path_with_tangents(self, path, tangents, scale=0.1):
        """Plot the path with tangents in a 3D plot."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(path[:, 0], path[:, 1], path[:, 2], label='Adjusted Path')

        for i in range(len(path)):
            x, y, z = path[i]
            dx, dy, dz = tangents[i] * scale
            ax.quiver(x, y, z, dx, dy, dz, color='r', length=scale, normalize=False)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

    def process_path(self, plot=True):
        """Process the path adjustments and optionally plot them. Return adjusted path and tangents."""
        adjusted_waypoints = self.adjust_path()
        adjusted_splines = self.fit_spline(adjusted_waypoints)
        adjusted_smooth_path = self.evaluate_spline(adjusted_splines)
        adjusted_tangents = self.compute_tangents(adjusted_splines)

        if plot:
            self.plot_3d_path_with_tangents(adjusted_smooth_path, adjusted_tangents, scale=0.05)

        return adjusted_smooth_path, adjusted_tangents


# Example usage (this should be outside the class in your main code block)
if __name__ == "__main__":
    # Example array of configurations
    # waypoints = np.array([[0, 0, 0], [1, 2, 0], [2, 4, 0], [3, 6, 0], [4, 8, 0], [5, 10, 0]])
    waypoints = create_sine_wave_waypoints()

    new_start = [5, -10, 0]
    new_end = [20, 0, 0]
    path_model = PathModel(waypoints, new_start, new_end)
    path_model.process_path()
