import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import distance

# Depot coordinates
depot = (44.651224, -63.611700)

# Number of delivery points
num_points = 100

# Maximum radius in kilometers
max_radius = 3

# Generate random angles and radii
angles = np.random.uniform(0, 2 * np.pi, num_points)  # Random distribution of angles
radii = np.sqrt(np.random.uniform(0, max_radius**2, num_points))  # Dense near center

# Calculate coordinates
coordinates = {}
lats = []
lons = []

for idx, (radius, angle) in enumerate(zip(radii, angles), start=1):
    # Calculate offset in lat/lon using geopy
    offset = distance(kilometers=radius).destination(depot, np.degrees(angle))
    coordinates[f'C{idx}'] = (offset.latitude, offset.longitude)
    lats.append(offset.latitude)
    lons.append(offset.longitude)

# Print coordinates in the desired format
for key, coord in coordinates.items():
    print(f"'{key}': {coord},")

# Plot the points
plt.figure(figsize=(10, 6))
plt.scatter(lons, lats, c='blue', label="Delivery Points")
plt.scatter(depot[1], depot[0], c='black', marker='*', s=100, label="Depot")
plt.title("Dense Circular Distribution of Delivery Points")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show()