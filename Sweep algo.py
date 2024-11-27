import numpy as np
import matplotlib.pyplot as plt
import math
import random
import copy
from collections import defaultdict
import seaborn as sns
import os
import pickle
import time 

start_time = time.time()  # Start timer

# Define Customer Demands and Coordinates
customers = { 
    'C1': 100, 'C2': 100, 'C3': 100, 'C4': 100, 'C5': 100,
    'C6': 200, 'C7': 200, 'C8': 200, 'C9': 200, 'C10': 200,
    'C11': 200, 'C12': 200, 'C13': 200, 'C14': 200, 'C15': 200,
    'C16': 50, 'C17': 50, 'C18': 50, 'C19': 50, 'C20': 50,
    'C21': 60, 'C22': 70, 'C23': 40, 'C24': 50, 'C25': 100,
    'C26': 90, 'C27': 80, 'C28': 110, 'C29': 120, 'C30': 130,
    'C31': 60, 'C32': 100, 'C33': 140, 'C34': 150, 'C35': 160,
    'C36': 70, 'C37': 90, 'C38': 130, 'C39': 60, 'C40': 75,
    'C41': 55, 'C42': 85, 'C43': 95, 'C44': 105, 'C45': 115,
    'C46': 125, 'C47': 135, 'C48': 145, 'C49': 155, 'C50': 165,  'C51': 195,
    'C52': 105,
    'C53': 155,
    'C54': 90,
    'C55': 205,
    'C56': 95,
    'C57': 145,
    'C58': 185,
    'C59': 95,
    'C60': 205,
    'C61': 155,
    'C62': 95,
    'C63': 195,
    'C64': 105,
    'C65': 155,
    'C66': 95,
    'C67': 205,
    'C68': 145,
    'C69': 95,
    'C70': 195,
    'C71': 105,
    'C72': 155,
    'C73': 95,
    'C74': 205,
    'C75': 145,
    'C76': 95,
    'C77': 195,
    'C78': 105,
    'C79': 155,
    'C80': 95,
    'C81': 205,
    'C82': 145,
    'C83': 95,
    'C84': 195,
    'C85': 105,
    'C86': 155,
    'C87': 95,
    'C88': 205,
    'C89': 145,
    'C90': 95,
    'C91': 195,
    'C92': 105,
    'C93': 155,
    'C94': 95,
    'C95': 205,
    'C96': 145,
    'C97': 95,
    'C98': 195,
    'C99': 105,
    'C100': 155
}

customer_coordinates = { 
'C1': (44.398166011511805, -64.15338793570588),
'C2': (44.404465527289375, -63.60295935985629),
'C3': (45.0908987549849, -64.11303222770913),
'C4': (44.533175696117674, -63.732487766489115),
'C5': (45.00109753513798, -63.934172036609944),
'C6': (45.0374211604903, -63.62007511588522),
'C7': (45.04568247697039, -63.15044525530558),
'C8': (44.45653825859552, -64.15267648329323),
'C9': (44.671490989032485, -63.386964914342634),
'C10': (45.06825883077847, -64.11077639452462),
'C11': (44.997296010123776, -63.242034348937906),
'C12': (44.4865791259699, -63.701902283660424),
'C13': (44.425284570781514, -64.0401049697032),
'C14': (44.44890405440713, -63.44570419357726),
'C15': (44.23565902663993, -63.13615128354227),
'C16': (44.43896880248532, -63.744460335398095),
'C17': (44.901561828532756, -63.33245672430979),
'C18': (44.381196357034334, -63.62858425124503),
'C19': (44.57444061542285, -63.64674804357786),
'C20': (44.80979992068303, -63.081276459864156),
'C21': (44.79456358024826, -63.0945016348679),
'C22': (44.724412795875175, -63.3979274760468),
'C23': (44.41155301499552, -63.68953982565029),
'C24': (45.00353300444, -63.55417833594273),
'C25': (44.51336280493192, -63.9353531909972),
'C26': (44.98999846955824, -64.23619932668586),
'C27': (44.45480091781568, -63.881318259625125),
'C28': (45.06658974212094, -63.18113749416233),
'C29': (44.226207345689446, -63.437074682338306),
'C30': (44.315458662199475, -64.05753623197559),
'C31': (44.962938542474845, -63.626693496169295),
'C32': (44.51332671855662, -63.021528992124864),
'C33': (44.314713555939555, -63.32326947724047),
'C34': (44.81550096296835, -63.3544901515585),
'C35': (44.30907125795768, -63.70955272388964),
'C36': (45.02503494843336, -63.904064547297644),
'C37': (45.02954843713219, -63.84464539690133),
'C38': (44.400982279751354, -63.70408834930041),
'C39': (44.22573590519709, -63.210545716393916),
'C40': (44.986650919960965, -63.73141139478341),
'C41': (44.850035086498146, -64.10838282819128),
'C42': (44.69925702694145, -63.04084372050852),
'C43': (44.586571067476434, -64.16040303908802),
'C44': (44.509454345598506, -63.09164789156158),
'C45': (44.38555089326621, -63.07585003625814),
'C46': (44.60135143225938, -63.70867625488085),
'C47': (44.305042354277695, -63.26925634192737),
'C48': (44.90950692527827, -63.41424752208317),
'C49': (44.33429222124697, -63.46286746561448),
'C50': (44.435570492385935, -63.514793630055024),
'C51': (44.84844722785863, -63.57725310475464),
'C52': (44.81260216220697, -63.40961154227853),
'C53': (44.26506014709823, -63.30389429819242),
'C54': (45.07766033616988, -63.369905031374714),
'C55': (44.360111273361134, -64.21163921471008),
'C56': (44.58939643374079, -63.07823285096813),
'C57': (44.55120992263921, -64.21606892889388),
'C58': (44.915949327576435, -64.23277740318684),
'C59': (44.48383301822828, -63.97119523170425),
'C60': (44.67107275268519, -63.61536196975914),
'C61': (45.074541882690134, -64.04874722262234),
'C62': (44.70969589182159, -63.620151519325375),
'C63': (44.664332079039134, -64.01996182249268),
'C64': (44.2385524397313, -64.12091550266251),
'C65': (44.605960898414665, -63.42476826703856),
'C66': (44.44149348480493, -63.107382137218835),
'C67': (44.40778484528735, -63.59668809936145),
'C68': (44.74696657630938, -63.63274048678967),
'C69': (44.796588268154366, -63.02579586919747),
'C70': (44.54864898971509, -63.51517376811067),
'C71': (44.46821425288743, -64.07847089088914),
'C72': (44.221496625971746, -63.92781441080624),
'C73': (44.6463823005442, -64.21934836688352),
'C74': (45.03934517547766, -64.13395650320896),
'C75': (44.48474617670563, -63.274661292034466),
'C76': (45.05486312784753, -63.183269706738635),
'C77': (44.713826565921, -63.163070058437114),
'C78': (44.37721642469655, -63.82017785376619),
'C79': (44.86033657132886, -63.359107839833115),
'C80': (44.92244119444311, -63.46260051020676),
'C81': (44.9392428005506, -63.694880841379344),
'C82': (44.31602472602713, -63.947368607018966),
'C83': (44.92351611904666, -63.03189401953439),
'C84': (44.42405958459692, -63.84441349124193),
'C85': (44.36554531285578, -63.16619789299426),
'C86': (44.84595622237252, -63.11790443934303),
'C87': (44.91604311707544, -64.07903230129344),
'C88': (44.533075289373905, -63.31269242330979),
'C89': (44.44445229941941, -63.25362460253641),
'C90': (44.899707404137615, -63.131546523088026),
'C91': (44.92615274758136, -63.85090792933522),
'C92': (44.796142456129, -63.5312847205521),
'C93': (44.63558501525755, -64.13848810373108),
'C94': (44.4375022949255, -63.18377614274586),
'C95': (44.82019647761486, -63.85212676157989),
'C96': (44.310937558475935, -64.06185866227109),
'C97': (44.64871898457204, -63.8389839279773),
'C98': (44.78277205366617, -63.79633097744338),
'C99': (44.222108024029545, -63.442955718380816),
'C100': (44.76990961552595, -63.521138952225556)
}

# Define Trucks with their capacities
trucks = {
    'A': 4500,
    'B': 4500,
    'C': 4500
}

# Define the depot coordinates
depot = (44.651224, -63.611700)

# 1. Convert Customer Coordinates to Polar Coordinates
def get_polar_coordinates(coordinate, depot):
    delta_x = coordinate[1] - depot[1]  # Longitude
    delta_y = coordinate[0] - depot[0]  # Latitude
    angle = math.atan2(delta_y, delta_x) * 180 / math.pi  # Convert to degrees
    angle = angle if angle >= 0 else angle + 360  # Ensure angle is between 0 and 360
    distance = math.sqrt(delta_x**2 + delta_y**2)  # Euclidean distance
    return angle, distance

# Calculate polar coordinates for all customers
customer_polar_coordinates = {}
for customer, coord in customer_coordinates.items():
    angle, distance = get_polar_coordinates(coord, depot)
    customer_polar_coordinates[customer] = (angle, distance)

# 2. Sort Customers by Angle (for the sweeping process)
sorted_customers = sorted(customer_polar_coordinates.items(), key=lambda x: x[1][0])

# 3. Sweep and Form Clusters Based on Truck Capacities
def form_clusters(customers, sorted_customers, trucks):
    clusters = []
    current_cluster = []
    current_demand = 0
    truck_capacities = list(trucks.values())
    truck_index = 0

    for customer, (angle, distance) in sorted_customers:
        demand = customers[customer]

        if current_demand + demand <= truck_capacities[truck_index]:
            current_cluster.append(customer)
            current_demand += demand
        else:
            clusters.append(current_cluster)
            truck_index += 1
            if truck_index >= len(truck_capacities):
                truck_index = 0
            current_cluster = [customer]
            current_demand = demand

    if current_cluster:
        clusters.append(current_cluster)

    return clusters

# Form clusters using the Sweep Algorithm
clusters = form_clusters(customers, sorted_customers, trucks)

# Assign trucks to clusters
truck_clusters = {}
for truck, cluster in zip(trucks.keys(), clusters):
    truck_clusters[truck] = cluster

# Display cluster coordinates and identify unsatisfied customers
unsatisfied_customers = set(customers.keys())
cluster_coordinates = {}

for truck, cluster in truck_clusters.items():
    cluster_coordinates[truck] = [customer_coordinates[customer] for customer in cluster]
    for customer in cluster:
        if customer in unsatisfied_customers:
            unsatisfied_customers.remove(customer)

# Include the depot in the coordinates for each truck
print("\nCoordinates of clusters formed by each truck (including depot):")
for truck, coordinates in cluster_coordinates.items():
    # Insert depot at the start and end of the coordinate list
    full_route = [depot] + coordinates + [depot]
    coord_list = ', '.join([f"({lat}, {lon})" for lat, lon in full_route])
    print(f"Truck {truck} full route coordinates: {coord_list}\n")

# Print unsatisfied customers if any
if unsatisfied_customers:
    print("Customers unsatisfied (unassigned to any truck due to capacity constraints):")
    for customer in unsatisfied_customers:
        coord = customer_coordinates[customer]
        print(f"{customer} with demand {customers[customer]} at coordinates ({coord[0]}, {coord[1]})")
else:
    print("All customers were successfully assigned to a truck.")

# Calculate and display the remaining capacities for each truck
print("\nRemaining capacities for each truck after cluster assignments:")
for truck in trucks:
    assigned_customers = truck_clusters.get(truck, [])
    total_demand = sum(customers[customer] for customer in assigned_customers)
    remaining_capacity = trucks[truck] - total_demand
    print(f"Truck {truck}: Remaining Capacity = {remaining_capacity}")

    end_time = time.time()  # End timer
    execution_time = end_time - start_time
    print(f"\nTime taken to run the code: {execution_time:.2f} seconds")

# 4. Visualize the Clusters and the Sweeping Process
def plot_sweep_clusters(truck_clusters, customer_coordinates, depot):
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    plt.figure(figsize=(10, 8))

    plt.scatter(depot[1], depot[0], color='black', marker='X', s=200, label='Depot')

    for i, (truck, customers) in enumerate(truck_clusters.items()):
        x_coords = [customer_coordinates[customer][0] for customer in customers]
        y_coords = [customer_coordinates[customer][1] for customer in customers]
        plt.scatter(y_coords, x_coords, color=colors[i % len(colors)], label=f'Truck {truck} Cluster')
        
        for customer in customers:
            plt.plot([depot[1], customer_coordinates[customer][1]], [depot[0], customer_coordinates[customer][0]], color=colors[i % len(colors)], linestyle='--')

    plt.title("Sweep Algorithm Clusters for Trucks")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.show()

plot_sweep_clusters(truck_clusters, customer_coordinates, depot)