import random

# ---- Step 1: Problem Setup ----
cities = list(range(6))  # 6 cities labeled 0 to 5
distance_matrix = [
    [0, 2, 9, 10, 7, 3],
    [2, 0, 6, 4, 3, 8],
    [9, 6, 0, 8, 5, 6],
    [10, 4, 8, 0, 6, 7],
    [7, 3, 5, 6, 0, 4],
    [3, 8, 6, 7, 4, 0]
]

# Parameters
POP_SIZE = 20
MUTATION_RATE = 0.1
GENERATIONS = 200

# ---- Step 2: Fitness Function ----
def total_distance(route):
    distance = 0
    for i in range(len(route)):
        distance += distance_matrix[route[i]][route[(i+1) % len(route)]]
    return distance

def fitness(route):
    return 1 / total_distance(route)

# ---- Step 3: Initialize Population ----
def create_route():
    route = cities[:]
    random.shuffle(route)
    return route

def initial_population():
    return [create_route() for _ in range(POP_SIZE)]

# ---- Step 4: Selection ----
def selection(population):
    population.sort(key=total_distance)
    return population[:POP_SIZE//2]  # take best half

# ---- Step 5: Crossover ---- (Order Crossover)
def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [None]*len(parent1)
    child[start:end] = parent1[start:end]

    fill_values = [gene for gene in parent2 if gene not in child]
    pos = 0
    for i in range(len(child)):
        if child[i] is None:
            child[i] = fill_values[pos]
            pos += 1
    return child

# ---- Step 6: Mutation ----
def mutate(route):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route

# ---- Step 7: Main Loop ----
def gene_expression_algorithm():
    population = initial_population()
    best_route = None
    best_distance = float('inf')

    for gen in range(GENERATIONS):
        selected = selection(population)
        children = []
        while len(children) < POP_SIZE:
            p1, p2 = random.sample(selected, 2)
            child = crossover(p1, p2)
            child = mutate(child)
            children.append(child)
        population = children

        # Track best
        current_best = min(population, key=total_distance)
        current_dist = total_distance(current_best)
        if current_dist < best_distance:
            best_route = current_best
            best_distance = current_dist

        if gen % 20 == 0:  # log progress
            print(f"Gen {gen}: Best Distance = {best_distance}")

    return best_route, best_distance

# ---- Run ----
best_route, best_distance = gene_expression_algorithm()
print("\nBest Route:", best_route)
print("Best Distance:", best_distance)
