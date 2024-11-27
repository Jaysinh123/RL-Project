import numpy as np
import random
import math
import matplotlib.pyplot as plt
import time  # Import time module to measure execution time

start_time = time.time()  # Start timer

# Define customer demands
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

# Define trucks with their capacities
trucks = {
    'A': 4500,
    'B': 4500,
    'C': 4500
}

# Define customer coordinates (latitude, longitude)
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

# Depot location
depot = (44.651224, -63.611700)

# Function to calculate the angle between the depot and a customer
def calculate_angle(coord1, coord2):
    delta_y = coord2[0] - coord1[0]
    delta_x = coord2[1] - coord1[1]
    angle = math.degrees(math.atan2(delta_y, delta_x))
    return angle % 360

# Assign customers to clusters based on the starting reference angle, considering truck capacities
def assign_customers_to_initial_sectors(customer_coordinates, depot, num_trucks, truck_capacities, demands, theta_start=0):
    angle_per_sector = 360 / num_trucks
    clusters = {i: [] for i in range(num_trucks)}  # Create empty clusters for each truck
    capacities_left = truck_capacities.copy()  # Track remaining capacity per truck

    # Calculate angles for each customer relative to the depot
    customer_angles = []
    for customer, coordinates in customer_coordinates.items():
        angle = (calculate_angle(depot, coordinates) - theta_start) % 360  # Shift the angle by theta_start
        customer_angles.append((angle, customer))

    # Sort customers by angle to assign them to sectors
    customer_angles.sort()

    # Assign customers to clusters while checking capacity constraints
    for angle, customer in customer_angles:
        sector = int(angle // angle_per_sector)
        demand = demands[customer]

        # Assign the customer if there is enough capacity in the truck
        if capacities_left[sector] >= demand:
            clusters[sector].append(customer)
            capacities_left[sector] -= demand
        else:
            # Attempt to find another truck that can handle the demand
            assigned = False
            for i in range(num_trucks):
                if capacities_left[i] >= demand:
                    clusters[i].append(customer)
                    capacities_left[i] -= demand
                    assigned = True
                    break
            if not assigned:
                print(f"Warning: Cannot assign customer {customer} due to capacity constraints.")

    return clusters, capacities_left

def plot_clusters(clusters, customer_coordinates, depot, title):
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    plt.figure(figsize=(10, 8))

    # Plot each cluster with a different color
    for cluster_id, customers in clusters.items():
        if customers:
            xs = [customer_coordinates[c][1] for c in customers]  # Longitude
            ys = [customer_coordinates[c][0] for c in customers]  # Latitude
            plt.scatter(xs, ys, color=colors[cluster_id % len(colors)], label=f'Truck {cluster_id + 1}')
    # Plot the depot
    plt.scatter(depot[1], depot[0], color='black', marker='*', s=200, label='Depot')
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True)
    plt.show()


# Main function to run the clustering
if __name__ == "__main__":

    num_trucks = len(trucks)
    truck_capacities = [trucks[key] for key in trucks]
    customer_demands = customers  # Ensure customer demands are correctly defined

    # Step 1: Initial clustering based purely on angular division with capacities
    initial_clusters, capacities_left = assign_customers_to_initial_sectors(
        customer_coordinates, depot, num_trucks, truck_capacities.copy(), customer_demands
    )

    # Print the clusters formed by initial angular division
    print("\nCluster Assignments by Initial Angular Division:")
    for cluster_id, customers_in_cluster in initial_clusters.items():
        print(f"Truck {cluster_id + 1}: {customers_in_cluster}")

    # Prepare to output coordinates including depot
    print("\nCoordinates of Each Cluster (including depot):")
    for cluster_id, customers_in_cluster in initial_clusters.items():
        print(f"\nTruck {cluster_id + 1} Coordinates (including depot):")
        coordinates = [depot]  # Start with depot
        for customer in customers_in_cluster:
            coord = customer_coordinates[customer]
            coordinates.append((round(coord[0], 6), round(coord[1], 6)))
        coordinates.append(depot)  # End with depot
        print(f"  coordinates = {coordinates}")

    # Calculate and print the remaining capacities
    print("\nTruck Capacities Left After Initial Angular Division:")
    for cluster_id in initial_clusters.keys():
        remaining_capacity = capacities_left[cluster_id]
        print(f"Truck {cluster_id + 1}: Remaining Capacity = {remaining_capacity}")

    # # Brief explanation of remaining truck capacities
    # print("\nInterpretation of Remaining Truck Capacities:")
    # print("The remaining capacity for each truck indicates how much more load the truck can carry.")
    # print("A negative remaining capacity means the truck is overloaded by that amount.")
    # print("A positive remaining capacity means the truck is underutilized by that amount.")
    # print("Ideally, all trucks should have remaining capacities close to zero to maximize efficiency and balance the load.")

    end_time = time.time()  # End timer
    execution_time = end_time - start_time
    print(f"\nTime taken to run the code: {execution_time:.2f} seconds")

# Call the function to plot the initial clusters
plot_clusters(
    initial_clusters, customer_coordinates, depot,
    "Clusters Formed by Initial Angular Division"
)
