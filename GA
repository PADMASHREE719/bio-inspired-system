import random
import math

# 1. Define the fitness function
def fitness(x):
    return x * math.sin(10 * math.pi * x) + 1.0  # x ∈ [0,1]

# 2. Parameters
POP_SIZE = 50
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.9
GENERATIONS = 100

# 3. Create initial population (x in [0, 1])
def create_population(size):
    return [random.uniform(0, 1) for _ in range(size)]

# 4. Evaluate fitness of population
def evaluate_population(population):
    return [fitness(ind) for ind in population]

# 5. Select two parents via tournament selection
def select_parents(population, fitnesses):
    def tournament():
        i, j = random.sample(range(len(population)), 2)
        return population[i] if fitnesses[i] > fitnesses[j] else population[j]
    return tournament(), tournament()

# 6. Crossover: Blend two parents
def crossover(p1, p2):
    if random.random() < CROSSOVER_RATE:
        alpha = random.random()
        return alpha * p1 + (1 - alpha) * p2
    return p1  # No crossover

# 7. Mutation: Small random change
def mutate(individual):
    if random.random() < MUTATION_RATE:
        mutation_strength = 0.1
        individual += random.uniform(-mutation_strength, mutation_strength)
        individual = max(0, min(1, individual))  # Ensure within [0,1]
    return individual

# 8. Run GA
def genetic_algorithm():
    population = create_population(POP_SIZE)
    best_individual = None
    best_fitness = -float('inf')

    for generation in range(GENERATIONS):
        fitnesses = evaluate_population(population)
       
        # Track best individual
        for i in range(len(population)):
            if fitnesses[i] > best_fitness:
                best_fitness = fitnesses[i]
                best_individual = population[i]
       
        # Create new generation
        new_population = []
        while len(new_population) < POP_SIZE:
            parent1, parent2 = select_parents(population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
       
        population = new_population

    print(f"Best solution: x = {best_individual:.5f}, f(x) = {best_fitness:.5f}")

# 9. Run the algorithm
genetic_algorithm()
