import random

# ---------------------------
# Cost matrix (task-resource)
# ---------------------------
C = [
    [8, 6, 10],  # Task 1 costs
    [7, 5, 9],   # Task 2 costs
    [9, 8, 4]    # Task 3 costs
]

num_tasks = len(C)
num_resources = len(C[0])

# ---------------------------
# PCA Parameters
# ---------------------------
M, N = 3, 3           # Grid size
max_iter = 5           # Number of iterations
p_mut = 0.1            # Mutation probability

# ---------------------------
# Helper Functions
# ---------------------------
def compute_cost(allocation):
    """Compute total cost for a given allocation."""
    return sum(C[i][allocation[i]] for i in range(num_tasks))

def random_allocation():
    """Generate a random valid allocation (permutation)."""
    alloc = list(range(num_resources))
    random.shuffle(alloc)
    return alloc

def random_swap(alloc):
    """Randomly swap two tasks' resource assignments."""
    a = alloc[:]
    i, j = random.sample(range(num_tasks), 2)
    a[i], a[j] = a[j], a[i]
    return a

def swap_first_difference(a, b):
    """Move a slightly toward b by swapping the first differing task."""
    a = a[:]
    for i in range(num_tasks):
        if a[i] != b[i]:
            # Find index where b[i] currently exists in a
            j = a.index(b[i])
            a[i], a[j] = a[j], a[i]
            break
    return a

def get_neighbors(grid, x, y):
    """Return Moore neighborhood of (x,y)."""
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < M and 0 <= ny < N:
                neighbors.append(grid[nx][ny])
    return neighbors

# ---------------------------
# Initialize grid
# ---------------------------
grid = []
for i in range(M):
    row = []
    for j in range(N):
        alloc = random_allocation()
        cost = compute_cost(alloc)
        row.append({"alloc": alloc, "cost": cost})
    grid.append(row)

# Track global best
best_cell = min((grid[i][j] for i in range(M) for j in range(N)), key=lambda c: c["cost"])
best_alloc = best_cell["alloc"][:]
best_cost = best_cell["cost"]

# ---------------------------
# Main PCA Loop
# ---------------------------
for iteration in range(1, max_iter + 1):
    new_grid = [[cell.copy() for cell in row] for row in grid]

    for i in range(M):
        for j in range(N):
            cell = grid[i][j]
            neighbors = get_neighbors(grid, i, j)

            # Find best neighbor (tie-break randomly)
            min_cost = min(n["cost"] for n in neighbors)
            best_neighbors = [n for n in neighbors if n["cost"] == min_cost]
            chosen = random.choice(best_neighbors)

            # Move slightly toward best neighbor
            candidate = swap_first_difference(cell["alloc"], chosen["alloc"])
            candidate_cost = compute_cost(candidate)

            # Mutation
            if random.random() < p_mut:
                candidate = random_swap(candidate)
                candidate_cost = compute_cost(candidate)

            # Accept if better
            if candidate_cost < cell["cost"]:
                new_grid[i][j] = {"alloc": candidate, "cost": candidate_cost}

    # Update grid
    grid = new_grid

    # Update global best
    current_best = min((grid[i][j] for i in range(M) for j in range(N)), key=lambda c: c["cost"])
    if current_best["cost"] < best_cost:
        best_cost = current_best["cost"]
        best_alloc = current_best["alloc"][:]

    # Print iteration summary
    print(f"\nIteration {iteration}:")
    for r in range(M):
        for c in range(N):
            print(grid[r][c]["alloc"], "->", grid[r][c]["cost"], end="   ")
        print()
    print("Best so far:", best_alloc, "Cost:", best_cost)

# ---------------------------
# Final Result
# ---------------------------
print("\nFinal Best Allocation:", best_alloc)
print("Minimum Total Cost:", best_cost)
