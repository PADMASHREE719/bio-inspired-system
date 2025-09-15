import numpy as np
import math

# --- Levy flight step function ---
def levy_flight(Lambda):
    # Standard formula for Levy distribution step length
    sigma = (math.gamma(1 + Lambda) * math.sin(math.pi * Lambda / 2) /
            (math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
    
    u = np.random.normal(0, sigma)
    v = np.random.normal(0, 1)
    step = u / (abs(v) ** (1 / Lambda))
    return step

# --- Objective Function (example: Sphere function) ---
def objective(x):
    return np.sum(x**2)

# --- Generate new solution using Levy flights ---
def get_cuckoo(nest, best, step_size=0.01):
    s = nest + step_size * levy_flight(1.5) * (nest - best)
    return s

# --- Simple bounds handling ---
def simple_bounds(s, lower_bound, upper_bound):
    s = np.clip(s, lower_bound, upper_bound)
    return s

# --- Cuckoo Search Algorithm ---
def cuckoo_search(n=25, dim=2, lower_bound=-5, upper_bound=5, max_iter=1000, pa=0.25):
    # Initialize nests randomly
    nests = np.random.uniform(low=lower_bound, high=upper_bound, size=(n, dim))
    fitness = np.apply_along_axis(objective, 1, nests)
    
    best_index = np.argmin(fitness)
    best = nests[best_index].copy()
    best_fitness = fitness[best_index]

    for t in range(max_iter):
        # Get new cuckoo solution by levy flight
        i = np.random.randint(n)
        new_sol = get_cuckoo(nests[i], best)
        new_sol = simple_bounds(new_sol, lower_bound, upper_bound)
        new_fitness = objective(new_sol)

        # If new solution is better, replace it
        if new_fitness < fitness[i]:
            nests[i] = new_sol
            fitness[i] = new_fitness

        # Choose a random nest to compare and maybe replace
        j = np.random.randint(n)
        if fitness[i] < fitness[j]:
            nests[j] = nests[i].copy()
            fitness[j] = fitness[i]

        # Abandon some nests with probability pa
        K = np.random.rand(n, dim) > pa
        stepsize = np.random.rand() * (nests[np.random.permutation(n)] - nests[np.random.permutation(n)])
        new_nests = nests + stepsize * K
        new_nests = np.clip(new_nests, lower_bound, upper_bound)
        new_fitness = np.apply_along_axis(objective, 1, new_nests)

        # Replace nests if new ones are better
        mask = new_fitness < fitness
        nests[mask] = new_nests[mask]
        fitness[mask] = new_fitness[mask]

        # Update best
        best_index = np.argmin(fitness)
        if fitness[best_index] < best_fitness:
            best = nests[best_index].copy()
            best_fitness = fitness[best_index]

        # Print progress every 100 iterations
        if t % 100 == 0:
            print(f"Iteration {t}: Best = {best}, Fitness = {best_fitness}")

    return best, best_fitness

# --- Run Example ---
if __name__ == "__main__":
    best_sol, best_val = cuckoo_search()
    print("\nFinal Best Solution:", best_sol)
    print("Final Best Fitness:", best_val)
