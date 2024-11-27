import random
import math
import matplotlib.pyplot as plt

# Depot location (latitude, longitude)
depot_lat, depot_lon = 44.651224, -63.611700

# Distance range (in kilometers)
inner_radius_km = 3  # Inner radius
outer_radius_km = 20  # Outer radius

# Earth's radius in kilometers
earth_radius_km = 6371

# Function to generate random points within the donut range
def generate_donut_coordinates(depot_lat, depot_lon, inner_radius_km, outer_radius_km, num_points):
    coordinates = []
    for _ in range(num_points):
        while True:
            # Generate a random point within a 20 km square around the depot
            delta_lat = random.uniform(-outer_radius_km / 111, outer_radius_km / 111)
            delta_lon = random.uniform(-outer_radius_km / (111 * math.cos(math.radians(depot_lat))),
                                        outer_radius_km / (111 * math.cos(math.radians(depot_lat))))
            lat = depot_lat + delta_lat
            lon = depot_lon + delta_lon

            # Calculate the distance from the depot using the Haversine formula
            lat1, lon1, lat2, lon2 = map(math.radians, [depot_lat, depot_lon, lat, lon])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            distance_km = earth_radius_km * c

            # Check if the point lies within the donut range
            if inner_radius_km <= distance_km <= outer_radius_km:
                coordinates.append((round(lat, 6), round(lon, 6)))
                break
    return coordinates

# Generate 100 donut-shaped coordinates
donut_coordinates = generate_donut_coordinates(depot_lat, depot_lon, inner_radius_km, outer_radius_km, 100)

# Format the coordinates in the desired dictionary format
formatted_output = {}
for i, coordinate in enumerate(donut_coordinates, start=1):
    formatted_output[f'C{i}'] = coordinate

# Print the coordinates in the required format
for key, value in formatted_output.items():
    print(f"    '{key}': {value},")

# Separate the latitudes and longitudes for plotting
lats, lons = zip(*donut_coordinates)

# Plot the points
plt.figure(figsize=(10, 6))
plt.scatter(lons, lats, c='blue', label="Delivery Points")  # Delivery points
plt.scatter(depot_lon, depot_lat, c='black', marker='*', s=100, label="Depot")  # Depot marker
plt.title("Distribution of Delivery Points over Donut-Shaped Area")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(loc='best')

# Show the plot
plt.show()


