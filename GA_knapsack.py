import random

# -------------------------------
# Problem Setup
# -------------------------------
items = [
    (2, 6),   # (weight, value)
    (3, 10),
    (4, 12),
    (5, 18),
    (9, 22)
]
capacity = 10
POP_SIZE = 6      # Number of chromosomes
GENS = 15         # Number of generations
MUT_RATE = 0.1    # Mutation probability

# -------------------------------
# Helper Functions
# -------------------------------

def fitness(chromosome):
    weight, value = 0, 0
    for gene, (w, v) in zip(chromosome, items):
        if gene == 1:
            weight += w
            value += v
    if weight > capacity:
        return 0  # Invalid solution
    return value

def create_chromosome():
    return [random.randint(0, 1) for _ in range(len(items))]

def selection(pop, fits):
    # Roulette Wheel Selection
    total_fit = sum(fits)
    if total_fit == 0:
        return random.choice(pop)
    pick = random.uniform(0, total_fit)
    current = 0
    for chrom, fit in zip(pop, fits):
        current += fit
        if current >= pick:
            return chrom

def crossover(p1, p2):
    point = random.randint(1, len(items)-1)
    return p1[:point] + p2[point:], p2[:point] + p1[point:]

def mutate(chrom):
    for i in range(len(chrom)):
        if random.random() < MUT_RATE:
            chrom[i] = 1 - chrom[i]  # Flip bit
    return chrom

# -------------------------------
# Genetic Algorithm
# -------------------------------

population = [create_chromosome() for _ in range(POP_SIZE)]

for gen in range(GENS):
    fitnesses = [fitness(c) for c in population]
    best_fit = max(fitnesses)
    best_chrom = population[fitnesses.index(best_fit)]
    print(f"Generation {gen+1}: Best={best_chrom} Value={best_fit}")

    new_population = []
    while len(new_population) < POP_SIZE:
        parent1 = selection(population, fitnesses)
        parent2 = selection(population, fitnesses)
        child1, child2 = crossover(parent1, parent2)
        new_population.append(mutate(child1))
        if len(new_population) < POP_SIZE:
            new_population.append(mutate(child2))
    population = new_population

# Final Result
fitnesses = [fitness(c) for c in population]
best_fit = max(fitnesses)
best_chrom = population[fitnesses.index(best_fit)]

print("\nBest Solution Found:")
print(f"Chromosome: {best_chrom}")
print(f"Total Value: {best_fit}")
print(f"Total Weight: {sum(w for gene,(w,v) in zip(best_chrom,items) if gene)}")
