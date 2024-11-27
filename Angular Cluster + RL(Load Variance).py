import numpy as np
import random
import math
import matplotlib.pyplot as plt
import pickle
import time
from collections import defaultdict
import os


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

start_time = time.time()  # Start timer

class RLClusterAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9,
                 epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.1):
        self.q_table = {}
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.theta_values = list(range(0, 360, 5))  # Discrete actions every 5 degrees

    def select_action(self):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.theta_values)
        else:
            if self.q_table:
                max_q = max(self.q_table.values())
                max_actions = [action for action, q in self.q_table.items() if q == max_q]
                return random.choice(max_actions)
            else:
                return random.choice(self.theta_values)

    def update_q_value(self, action, reward):
        action = float(action)  # Ensure action is a float
        current_q = self.q_table.get(action, 0)
        max_future_q = max(self.q_table.values(), default=0)
        new_q = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)
        self.q_table[action] = new_q

    def calculate_reward(self, clusters, demands, truck_capacities):
        load_per_truck = [sum(demands[c] for c in clusters[i]) for i in clusters]
        avg_load = sum(load_per_truck) / len(load_per_truck)
        load_deviation = sum(abs(load - avg_load) for load in load_per_truck)

        # Capacity violation penalty
        capacity_penalty = sum(
            max(0, load - capacity) * 1000
            for load, capacity in zip(load_per_truck, truck_capacities)
        )

        reward = -load_deviation - capacity_penalty
        return reward

    def train(self, customer_coordinates, demands, depot, num_trucks, truck_capacities, episodes=150, steps_per_episode=100):
        best_theta = 0
        best_clusters = None
        best_reward = float('-inf')

        for episode in range(episodes):
            for step in range(steps_per_episode):
                # Select an action (theta_start)
                theta_start = self.select_action()

                # Assign customers to clusters based on the current theta_start
                clusters = self.assign_customers_to_sectors(
                    customer_coordinates, depot, num_trucks, truck_capacities.copy(), demands, theta_start
                )

                # Calculate reward based on load deviation and capacity penalties
                reward = self.calculate_reward(clusters, demands, truck_capacities)

                # Update Q-value
                self.update_q_value(theta_start, reward)

                # Update best solution found so far
                if reward > best_reward:
                    best_reward = reward
                    best_theta = theta_start
                    best_clusters = clusters

            # Decay epsilon for exploration
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay

            # Print progress every 50 episodes
            if (episode + 1) % 50 == 0:
                print(f"Episode {episode + 1}/{episodes} - Best Reward: {best_reward}")

        return best_theta, best_clusters

    def assign_customers_to_sectors(self, customer_coordinates, depot, num_trucks, truck_capacities, demands, theta_start):
        theta_start = float(theta_start)  # Ensure theta_start is a float

        total_demand = sum(demands.values())
        demand_per_truck = total_demand / num_trucks
        sectors = {i: [] for i in range(num_trucks)}
        capacities_left = truck_capacities.copy()

        # Convert customer coordinates to NumPy arrays
        customer_coords_array = np.array([customer_coordinates[c] for c in customer_coordinates])
        depot_array = np.array(depot)

        # Calculate angles using vectorized operations
        delta_y = customer_coords_array[:, 0] - depot_array[0]
        delta_x = customer_coords_array[:, 1] - depot_array[1]
        angles = np.degrees(np.arctan2(delta_y, delta_x)) % 360
        adjusted_angles = (angles - theta_start) % 360

        # Combine angles with customer IDs
        customer_ids = list(customer_coordinates.keys())
        customer_angles = list(zip(adjusted_angles, customer_ids))
        customer_angles.sort()

        # Assign customers to sectors based on cumulative demand
        current_truck = 0
        cumulative_demand = 0
        for angle, customer in customer_angles:
            demand = demands[customer]
            if cumulative_demand + demand > demand_per_truck and current_truck < num_trucks - 1:
                current_truck += 1
                cumulative_demand = 0
            if capacities_left[current_truck] >= demand:
                sectors[current_truck].append(customer)
                capacities_left[current_truck] -= demand
                cumulative_demand += demand
            else:
                # Try to assign to another truck
                assigned = False
                for i in range(num_trucks):
                    if capacities_left[i] >= demand:
                        sectors[i].append(customer)
                        capacities_left[i] -= demand
                        assigned = True
                        break
                if not assigned:
                    print(f"Warning: Cannot assign customer {customer} due to capacity constraints.")

        return sectors

    def save_q_table(self, filename):
        with open(filename, 'wb') as f:
            # Convert keys to floats before saving
            q_table_to_save = {float(k): v for k, v in self.q_table.items()}
            pickle.dump(q_table_to_save, f)
        print(f"Q-table saved to {filename}")

    def load_q_table(self, filename):
        try:
            with open(filename, 'rb') as f:
                loaded_q_table = pickle.load(f)
                # Ensure that keys are floats
                self.q_table = {float(k): v for k, v in loaded_q_table.items()}
            print(f"Q-table loaded from {filename}")
        except (FileNotFoundError, EOFError, pickle.UnpicklingError, TypeError, ValueError):
            print("No previous Q-table found or file is corrupted. Starting fresh.")
            self.q_table = {}

    def print_q_table(self):
    # Print the Q-table in a readable format
        print("Q-Table (Theta Start vs Q-Value):")
    
    # Print in lines of 3 values per row
        for i, theta in enumerate(self.theta_values):
            q_value = self.q_table.get(theta, 0)  # Default to 0 if no value is found for theta
            print(f"Theta Start: {theta}°, Q-Value: {q_value:.4f}", end="   ")
        
        # Add line break after every 3rd value
            if (i + 1) % 3 == 0:
                print()  # Start a new line after every 3rd value

# Main function to run the RL agent
if __name__ == "__main__":
    # Delete the old Q-table file if it exists
    if os.path.exists('q_table.pkl'):
        os.remove('q_table.pkl')
        print("Old Q-table file deleted.")

# Function to calculate the angle between the depot and a customer
def calculate_angle(coord1, coord2):
    delta_y = coord2[0] - coord1[0]
    delta_x = coord2[1] - coord1[1]
    angle = math.degrees(math.atan2(delta_y, delta_x))
    return angle % 360

# Function to print cluster coordinates
def print_cluster_coordinates(best_clusters, customer_coordinates, depot):
    print("\nCoordinates of Each Cluster After RL Optimization (including depot):")
    for cluster_id, customers_in_cluster in best_clusters.items():
        print(f"\nTruck {cluster_id + 1} Coordinates (including depot):")
        coordinates = [depot]  # Start with depot
        for customer in customers_in_cluster:
            coord = customer_coordinates[customer]
            coordinates.append((round(coord[0], 6), round(coord[1], 6)))
        coordinates.append(depot)  # End with depot
        print(f"  coordinates = {coordinates}")

# Function to plot clusters
def plot_clusters(best_clusters, customer_coordinates, depot, best_theta):
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    plt.figure(figsize=(10, 8))

    # Plot each cluster
    for cluster_id, customers in best_clusters.items():
        if customers:
            xs = [customer_coordinates[c][1] for c in customers]  # Longitude
            ys = [customer_coordinates[c][0] for c in customers]  # Latitude
            plt.scatter(xs, ys, color=colors[cluster_id % len(colors)], label=f'Cluster {cluster_id + 1}')

    # Plot depot
    plt.scatter(depot[1], depot[0], color='black', marker='*', s=200, label='Depot')

    plt.title(f"Clusters Optimized by RL (Theta Start {best_theta:.2f} Degrees)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function to run the RL agent
if __name__ == "__main__":
    num_trucks = len(trucks)
    truck_capacities = list(trucks.values())

    agent = RLClusterAgent()

    # Load previous Q-table if available
    agent.load_q_table('q_table.pkl')

    # Train RL agent to optimize clustering based on load deviation and capacities
    best_theta, best_clusters = agent.train(
        customer_coordinates, customers, depot, num_trucks, truck_capacities.copy(), episodes=150, steps_per_episode=100
    )

    # Save the Q-table after training
    agent.save_q_table('q_table.pkl')

    end_time = time.time()  # End timer
    execution_time = end_time - start_time
    print(f"\nTime taken to run the code: {execution_time:.2f} seconds")

    # Print the best starting angle and clusters
    print(f"\nBest Theta Start: {best_theta:.2f} degrees")
    print("Best Cluster Assignments:")
    for cluster_id, customers_in_cluster in best_clusters.items():
        print(f"Truck {cluster_id + 1}: {customers_in_cluster}")

    # Print coordinates of each cluster in the requested format
    print_cluster_coordinates(best_clusters, customer_coordinates, depot)

    # Plot the clusters
    plot_clusters(best_clusters, customer_coordinates, depot, best_theta)

# Call this function to print the Q-table after training or loading
agent.print_q_table()
