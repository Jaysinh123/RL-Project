import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import distance

# Depot coordinates
depot = (44.651224, -63.611700)

# Number of delivery points
num_points = 100

# Maximum offset in kilometers
max_offset = 50  # +/-50 km in both latitude and longitude

# Generate random offsets in east and north directions
dxs = np.random.uniform(-max_offset, max_offset, num_points)
dys = np.random.uniform(-max_offset, max_offset, num_points)

# Calculate coordinates
coordinates = {}
lats = []
lons = []

for idx, (dx, dy) in enumerate(zip(dxs, dys), start=1):
    # Calculate distance and bearing from offsets
    distance_km = np.sqrt(dx**2 + dy**2)
    bearing = (np.degrees(np.arctan2(dx, dy))) % 360  # Note: arctangent2(dx, dy)
    
    # Calculate destination point
    offset = distance(kilometers=distance_km).destination(depot, bearing)
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
plt.title("Sparse Distribution of Delivery Points over 100x100 km Area")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show()