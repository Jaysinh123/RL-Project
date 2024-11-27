import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pickle
import os
import time 

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

# 4. Capacitated K-means Clustering
def calculate_distance(coord1, coord2):
    return math.hypot(coord1[0] - coord2[0], coord1[1] - coord2[1])

def initialize_centroids(coords, n_clusters):
    centroids = [depot] + random.sample(coords, n_clusters - 1)
    return centroids

def assign_customers_to_clusters(centroids, coords, demands, capacities, max_distance_threshold=float('inf')):
    clusters = {i: [] for i in range(len(centroids))}
    cluster_demands = {i: 0 for i in range(len(centroids))}

    unassigned_customers = set(coords.keys())

    while unassigned_customers:
        customer_assigned = False
        for customer in list(unassigned_customers):
            coord = coords[customer]
            min_distance = float('inf')
            best_cluster = None

            for i, centroid in enumerate(centroids):
                distance = calculate_distance(coord, centroid)
                if distance < min_distance and cluster_demands[i] + demands[customer] <= capacities[i]:
                    min_distance = distance
                    best_cluster = i

            if best_cluster is not None:
                clusters[best_cluster].append(customer)
                cluster_demands[best_cluster] += demands[customer]
                unassigned_customers.remove(customer)
                customer_assigned = True

        if not customer_assigned:
            break

    return clusters, cluster_demands

def update_centroids(clusters, coords):
    new_centroids = []
    for cluster in clusters.values():
        if cluster:
            cluster_coords = np.array([coords[customer] for customer in cluster])
            new_centroids.append(np.mean(cluster_coords, axis=0))
        else:
            new_centroids.append(depot)
    return new_centroids

def capacitated_kmeans(coords, demands, capacities, n_clusters, max_iter=10):
    coords_list = list(coords.values())
    centroids = initialize_centroids(coords_list, n_clusters)

    for _ in range(max_iter):
        clusters, cluster_demands = assign_customers_to_clusters(centroids, coords, demands, capacities)
        new_centroids = update_centroids(clusters, coords)

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return clusters, cluster_demands, centroids

truck_ids = list(trucks.keys())
truck_capacities = [trucks[tid] for tid in truck_ids]

clusters, cluster_demands, centroids = capacitated_kmeans(
    customer_coordinates, customers, truck_capacities, len(trucks)
)

cluster_id_to_truck = {cluster_id: truck_id for cluster_id, truck_id in enumerate(truck_ids)}
truck_id_to_index = {truck_id: idx for idx, truck_id in enumerate(truck_ids)}

# # 5. Q-Learning Agent with Modified Reward Function
# class QLearningAgent:
#     def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.01):
#         self.q_table = {}
#         self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.epsilon_decay = epsilon_decay
#         self.min_epsilon = min_epsilon

#     def select_action(self, state, valid_actions):
#         if not valid_actions:
#             return None
#         if random.uniform(0, 1) < self.epsilon:
#             return random.choice(valid_actions)
#         else:
#             q_values = [self.q_table.get((state, action), 0) for action in valid_actions]
#             max_q = max(q_values)
#             max_actions = [action for action, q in zip(valid_actions, q_values) if q == max_q]
#             return random.choice(max_actions)

#     def update_q_value(self, state, action, reward, next_state, next_valid_actions):
#         current_q = self.q_table.get((state, action), 0)
#         if next_valid_actions:
#             max_future_q = max([self.q_table.get((next_state, a), 0) for a in next_valid_actions])
#         else:
#             max_future_q = 0
#         new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
#         self.q_table[(state, action)] = new_q

#     def save_q_table(self, filename):
#         with open(filename, 'wb') as f:
#             pickle.dump(self.q_table, f)

#     def load_q_table(self, filename):
#         if os.path.exists(filename):
#             with open(filename, 'rb') as f:
#                 self.q_table = pickle.load(f)
#         else:
#             self.q_table = {}

# def modified_reward_function(current_coord, new_coord, cluster_coords):
#     # Reward should minimize the distance and also incentivize tighter clusters
#     distance_penalty = calculate_distance(current_coord, new_coord)
#     if len(cluster_coords) > 0:
#         centroid = np.mean(cluster_coords, axis=0)
#         cluster_coherence_penalty = np.mean([calculate_distance(centroid, coord) for coord in cluster_coords])
#     else:
#         cluster_coherence_penalty = 0
#     # Combining both penalties - coherence should be more weighted
#     return -(distance_penalty + 2 * cluster_coherence_penalty)

# agent = QLearningAgent()
# q_table_filename = "q_table_modified.pkl"
# agent.load_q_table(q_table_filename)

# 6. Print Assigned Customers and Remaining Capacities for Capacitated K-means
for cluster_id, customers_list in clusters.items():
    truck_id = cluster_id_to_truck[cluster_id]
    remaining_capacity = truck_capacities[truck_id_to_index[truck_id]] - sum(customers[c] for c in customers_list)
    cluster_coords = [customer_coordinates[c] for c in customers_list]
    print(f"Truck {truck_id}:\n  Assigned Customers: {customers_list}\n  Remaining Capacity: {remaining_capacity}\n  Coordinates (including Depot): {[depot] + cluster_coords}\n")

# Print number of unsatisfied customers
unsatisfied_customers = set(customers.keys()) - set(customer for cluster in clusters.values() for customer in cluster)
print(f"Number of unsatisfied customers: {len(unsatisfied_customers)}")

# # Attempt to Reassign Unsatisfied Customers
# if unsatisfied_customers:
#     for customer in unsatisfied_customers:
#         for cluster_id, truck_id in cluster_id_to_truck.items():
#             remaining_capacity = truck_capacities[truck_id_to_index[truck_id]] - cluster_demands[cluster_id]
#             if remaining_capacity >= customers[customer]:
#                 clusters[cluster_id].append(customer)
#                 cluster_demands[cluster_id] += customers[customer]
#                 break

# # Print Final Assigned Customers After Reassignment
# for cluster_id, customers_list in clusters.items():
#     truck_id = cluster_id_to_truck[cluster_id]
#     remaining_capacity = truck_capacities[truck_id_to_index[truck_id]] - sum(customers[c] for c in customers_list)
#     cluster_coords = [customer_coordinates[c] for c in customers_list]
#     print(f"Final Truck {truck_id}:\n  Assigned Customers: {customers_list}\n  Remaining Capacity: {remaining_capacity}\n  Coordinates (including Depot): {[depot] + cluster_coords}\n")

# Print number of unsatisfied customers after reassignment
unsatisfied_customers = set(customers.keys()) - set(customer for cluster in clusters.values() for customer in cluster)
print(f"Number of unsatisfied customers after reassignment: {len(unsatisfied_customers)}")

end_time = time.time()  # End timer
execution_time = end_time - start_time
print(f"\nTime taken to run the code: {execution_time:.2f} seconds")

# # 7. Q-Learning Algorithm
# num_episodes = 500  # Reduced from 500 to 100
# max_steps = 100  # Reduced from 50 to 30

# for episode in range(num_episodes):
#     state = (None, tuple(truck_capacities))  # Simplified state
#     steps = 0  # Track the number of steps within the episode

#     assigned_customers = set()
#     unassigned_customers = set(customers.keys())
#     remaining_capacities = list(truck_capacities)

#     while unassigned_customers and steps < max_steps:
#         valid_actions = []
#         for cluster_id, customers_list in clusters.items():
#             truck_id = cluster_id_to_truck[cluster_id]
#             truck_index = truck_id_to_index[truck_id]
#             for customer in customers_list:
#                 if customer in unassigned_customers and remaining_capacities[truck_index] >= customers[customer]:
#                     valid_actions.append((customer, truck_id))

#         action = agent.select_action(state, valid_actions)
#         if action is None:
#             break  # No valid actions left

#         customer, truck_id = action
#         truck_index = truck_id_to_index[truck_id]

#         # Perform action: assign customer to truck
#         assigned_customers.add(customer)
#         unassigned_customers.remove(customer)
#         remaining_capacities[truck_index] -= customers[customer]

#         # Calculate reward based on distance and cluster coherence
#         current_coord = depot if steps == 0 else customer_coordinates[customer]
#         new_coord = customer_coordinates[customer]
#         cluster_coords = [customer_coordinates[c] for c in assigned_customers]
#         reward = modified_reward_function(current_coord, new_coord, cluster_coords)

#         # Update state and get next valid actions
#         next_state = (None, tuple(remaining_capacities))

#         next_valid_actions = [
#             (c, truck_id_next)
#             for cluster_id, customers_list in clusters.items()
#             for truck_id_next in trucks
#             for c in customers_list
#             if c in unassigned_customers and remaining_capacities[truck_id_to_index[truck_id_next]] >= customers[c]
#         ]

#         # Update Q-value and state
#         agent.update_q_value(state, action, reward, next_state, next_valid_actions)
#         state = next_state
#         steps += 1  # Increment step counter

#     # Update epsilon after each episode
#     if agent.epsilon > agent.min_epsilon:
#         agent.epsilon *= agent.epsilon_decay

#     # Print epsilon value after every 100th episode
#     if (episode + 1) % 100 == 0:
#         print(f"After Episode {episode + 1}, Epsilon: {agent.epsilon:.4f}")

# agent.save_q_table(q_table_filename)

# print("Training complete and modified Q-table saved.")

# 8. Visualization
# Bar Plot: Number of Customers Assigned to Each Truck
def plot_customers_per_truck(clusters):
    truck_names = truck_ids
    num_customers = [len(clusters[i]) for i in range(len(truck_names))]
    plt.figure(figsize=(8, 6))
    plt.bar(truck_names, num_customers)
    plt.xlabel("Trucks")
    plt.ylabel("Number of Customers")
    plt.title("Number of Customers Assigned to Each Truck")
    plt.show()


# Plot the bar chart
plot_customers_per_truck(clusters)

# # Q-Value Heatmap
# def plot_q_value_heatmap(agent):
#     customers_list = list(customers.keys())
#     trucks_list = truck_ids
#     n_customers = len(customers_list)
#     n_trucks = len(trucks_list)
#     q_matrix = np.zeros((n_customers, n_trucks))

#     for (state, action), q_value in agent.q_table.items():
#         if isinstance(action, tuple) and len(action) == 2:
#             customer, truck_id = action
#             if customer in customers_list and truck_id in trucks_list:
#                 i = customers_list.index(customer)
#                 j = trucks_list.index(truck_id)
#                 q_matrix[i, j] = q_value

#     plt.figure(figsize=(10, 8))
#     sns.heatmap(q_matrix, annot=True, cmap='viridis', xticklabels=trucks_list, yticklabels=customers_list)
#     plt.xlabel("Trucks")
#     plt.ylabel("Customers")
#     plt.title("Q-Value Heatmap")
#     plt.show()

# # Plot the Q-value heatmap
# plot_q_value_heatmap(agent)

# Cluster Formation Plot
def plot_cluster_formation(clusters, customer_coordinates, centroids):
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    plt.figure(figsize=(10, 8))

    for i, (cluster_id, customers_list) in enumerate(clusters.items()):
        cluster_coords = np.array([customer_coordinates[c] for c in customers_list])
        plt.scatter(cluster_coords[:, 1], cluster_coords[:, 0], color=colors[i % len(colors)], label=f'Truck {truck_ids[i]} Cluster')
        plt.scatter(centroids[i][1], centroids[i][0], color='black', marker='X', s=100, label=f'Centroid {i+1}')

    plt.scatter(depot[1], depot[0], color='black', marker='*', s=200, label='Depot')

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Cluster Formation of Customers Assigned to Trucks")
    plt.legend()
    plt.show()

# Plot the cluster formation
plot_cluster_formation(clusters, customer_coordinates, centroids)
