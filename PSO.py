import numpy as np

# Objective function
def f(position):
    x, y = position
    return x**2 + y**2

class Particle:
    def __init__(self, bounds):
        self.position = np.array([np.random.uniform(bounds[0][0], bounds[0][1]),
                                  np.random.uniform(bounds[1][0], bounds[1][1])])
        self.velocity = np.zeros_like(self.position)
        self.pbest_position = self.position.copy()
        self.pbest_value = float('inf')

    def update_velocity(self, gbest_position, w, c1, c2):
        r1, r2 = np.random.rand(2)
        cognitive = c1 * r1 * (self.pbest_position - self.position)
        social = c2 * r2 * (gbest_position - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self, bounds):
        self.position += self.velocity
        # Keep particle inside bounds
        for i in range(len(bounds)):
            self.position[i] = np.clip(self.position[i], bounds[i][0], bounds[i][1])

def pso(objective_function, bounds, num_particles, max_iter, w=0.5, c1=1.5, c2=1.5):
    swarm = [Particle(bounds) for _ in range(num_particles)]
    gbest_value = float('inf')
    gbest_position = None

    for iteration in range(max_iter):
        for particle in swarm:
            fitness = objective_function(particle.position)

            # Update personal best
            if fitness < particle.pbest_value:
                particle.pbest_value = fitness
                particle.pbest_position = particle.position.copy()

            # Update global best
            if fitness < gbest_value:
                gbest_value = fitness
                gbest_position = particle.position.copy()

        # Update velocity and position for each particle
        for particle in swarm:
            particle.update_velocity(gbest_position, w, c1, c2)
            particle.update_position(bounds)

        print(f"Iteration {iteration+1}/{max_iter}, Best Fitness: {gbest_value}")

    return gbest_position, gbest_value

# Parameters
bounds = [(-10, 10), (-10, 10)]  # Search space for x and y
num_particles = 30
max_iter = 50

best_pos, best_val = pso(f, bounds, num_particles, max_iter)
print(f"Best position: {best_pos}, Best value: {best_val}")
