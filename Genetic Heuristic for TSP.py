import random
import math
import matplotlib.pyplot as plt

# Define the Point class
class Point:
    def __init__(self, x, y, is_center=False):
        self.x = x  # Latitude
        self.y = y  # Longitude
        self.is_center = is_center

# Convert list of tuples to Point objects
def create_points(coordinate_list):
    return [Point(lat, lon) for lat, lon in coordinate_list]

# Calculate the Haversine distance between two geographic points
def haversine_distance(point1, point2):
    R = 6371.0  # Earth radius in kilometers

    lat1 = math.radians(point1.x)
    lon1 = math.radians(point1.y)
    lat2 = math.radians(point2.x)
    lon2 = math.radians(point2.y)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c  # Distance in kilometers

# Calculate the total distance of a given route
def total_distance(route, points):
    dist = 0
    for i in range(len(route)):
        dist += haversine_distance(points[route[i]], points[route[(i + 1) % len(route)]])
    return dist

# Generate the initial population of possible routes
def generate_population(pop_size, points):
    population = []
    for _ in range(pop_size):
        route = list(range(len(points)))
        random.shuffle(route)
        population.append(route)
    return population

# Evaluate the fitness of each route in the population
def evaluate_population(population, points):
    fitness_values = []
    for route in population:
        dist = total_distance(route, points)
        fitness_values.append(1.0 / dist)
    return fitness_values

# Select routes based on their fitness values
def selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    probabilities = [f / total_fitness for f in fitness_values]
    selected = random.choices(population, weights=probabilities, k=len(population))
    return selected

# Perform ordered crossover between two parent routes
def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None]*size
    child[start:end] = parent1[start:end]
    ptr = end
    for gene in parent2:
        if gene not in child:
            if ptr >= size:
                ptr = 0
            child[ptr] = gene
            ptr += 1
    return child

# Mutate a route by swapping two cities
def mutate(route, mutation_rate=0.01):
    for swapped in range(len(route)):
        if random.random() < mutation_rate:
            swap_with = int(random.random() * len(route))
            route[swapped], route[swap_with] = route[swap_with], route[swapped]
    return route

# Main genetic algorithm function
def genetic_algorithm(points, pop_size=100, elite_rate=0.2, mutation_rate=0.01, generations=2000):
    population = generate_population(pop_size, points)
    elite_size = int(pop_size * elite_rate)
    best_route = None
    best_distance = float('inf')

    for generation in range(generations):
        fitness_values = evaluate_population(population, points)
        sorted_population = [route for _, route in sorted(zip(fitness_values, population), reverse=True)]
        population = sorted_population[:elite_size]
        while len(population) < pop_size:
            parent1, parent2 = random.sample(sorted_population[:50], 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            population.append(child)
        current_best_distance = 1.0 / max(fitness_values)
        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_route = sorted_population[0]
        # Optional: Print progress every 50 generations
        if (generation + 1) % 200 == 0:
            print(f"Generation {generation + 1}, Best distance: {best_distance:.2f} km")
    return best_route, best_distance

# Drawing functions
def draw_circle(ax, point):
    color = '#0f0' if point.is_center else '#000'
    ax.plot(point.y, point.x, 'o', color=color, markersize=5)

def draw_lines(ax, route, points):
    lats = [points[i].x for i in route] + [points[route[0]].x]
    lons = [points[i].y for i in route] + [points[route[0]].y]
    ax.plot(lons, lats, color='#f00', linewidth=1)

def draw(points, best_route, best_distance, SALES_MEN):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_title(f"{len(points)} cities with {SALES_MEN} Salesman\nBest Distance: {best_distance:.2f} km")

    # Draw each point
    for point in points:
        draw_circle(ax, point)

    # Draw the optimal route
    if best_route and len(best_route) == len(points):
        draw_lines(ax, best_route, points)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Define points as coordinates directly, they will be converted to Point objects
    coordinates={(44.651224, -63.6117), (44.221496625971746, -63.92781441080624), (44.43896880248532, -63.744460335398095), (44.4865791259699, -63.701902283660424), (44.57444061542285, -63.64674804357786), (44.400982279751354, -63.70408834930041), (44.41155301499552, -63.68953982565029), (44.30907125795768, -63.70955272388964), (44.381196357034334, -63.62858425124503), (44.404465527289375, -63.60295935985629), (44.40778484528735, -63.59668809936145), (44.222108024029545, -63.442955718380816), (44.226207345689446, -63.437074682338306), (44.435570492385935, -63.514793630055024), (44.33429222124697, -63.46286746561448), (44.26506014709823, -63.30389429819242), (44.44890405440713, -63.44570419357726), (44.314713555939555, -63.32326947724047), (44.54864898971509, -63.51517376811067), (44.22573590519709, -63.210545716393916), (44.305042354277695, -63.26925634192737), (44.23565902663993, -63.13615128354227), (44.36554531285578, -63.16619789299426), (44.44445229941941, -63.25362460253641), (44.4375022949255, -63.18377614274586), (44.38555089326621, -63.07585003625814), (44.48474617670563, -63.274661292034466), (44.44149348480493, -63.107382137218835), (44.533075289373905, -63.31269242330979), (44.509454345598506, -63.09164789156158), (44.605960898414665, -63.42476826703856), (44.51332671855662, -63.021528992124864), (44.58939643374079, -63.07823285096813), (44.651224, -63.6117)
}

    # Convert coordinates to Point objects
    points = create_points(coordinates)

    # Parameters
    generations = 1600
    SALES_MEN = 1  # Adjust if implementing multiple salesmen

    # Run the genetic algorithm to solve the TSP
    best_route, best_distance = genetic_algorithm(points, generations=generations)

    # Draw the final result
    draw(points, best_route, best_distance, SALES_MEN)
