import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

# Input data points
x_data = np.array([129, 140, 103.5, 88, 185.5, 195, 105, 157.5, 107.5, 77, 81, 162, 162, 117.5])
y_data = np.array([7.5, 141.5, 23, 147, 22.5, 137.5, 85.5, -6.5, -81, 3, 56.5, -66.5, 84, -33.5])
z_data = -np.array([4, 8, 6, 8, 6, 8, 8, 9, 9, 8, 8, 9, 4, 9])

# Create a grid of points for interpolation
xi = np.linspace(75, 200, 200)
yi = np.linspace(-50, 200, 200)
xi, yi = np.meshgrid(xi, yi)

# Perform two-dimensional interpolation using the griddata function
# interpolate unstructured D-D data.
zi = griddata((x_data, y_data), z_data, (xi, yi), method='cubic')

# Create a three-dimensional plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the interpolated surface
ax.plot_surface(xi, yi, zi, cmap='viridis')
dangerous = -5*np.ones_like(zi)
ax.plot_surface(xi, yi, dangerous, color='Red')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Interpolated Surface')

# Display the plot
plt.show()