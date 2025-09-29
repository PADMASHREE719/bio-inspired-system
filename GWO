import numpy as np

def gwo(obj_func, lb, ub, dim, num_wolves=30, max_iter=100, seed=None):
    """
    Grey Wolf Optimizer (GWO) - minimization.

    Parameters
    ----------
    obj_func : callable
        Function to minimize. Accepts a 1D numpy array of length `dim` and returns scalar.
    lb, ub : float or array-like
        Lower and upper bounds (scalars or arrays of length dim).
    dim : int
        Dimensionality of the problem.
    num_wolves : int
        Population size.
    max_iter : int
        Number of iterations.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    best_pos : ndarray (dim,)
        Best solution found.
    best_score : float
        Objective value at best_pos.
    history : list of best_score per iteration
        Convergence history.
    """
    if seed is not None:
        np.random.seed(seed)

    lb = np.asarray(lb).reshape(dim) if np.shape(lb) not in ((), (dim,)) else np.full(dim, lb)
    ub = np.asarray(ub).reshape(dim) if np.shape(ub) not in ((), (dim,)) else np.full(dim, ub)

    # Initialize wolves positions uniformly within bounds
    wolves = np.random.uniform(lb, ub, (num_wolves, dim))
    fitness = np.array([obj_func(w) for w in wolves])

    # Initialize alpha, beta, delta (best, 2nd, 3rd)
    idx_sorted = np.argsort(fitness)
    alpha_pos = wolves[idx_sorted[0]].copy(); alpha_score = fitness[idx_sorted[0]]
    beta_pos  = wolves[idx_sorted[1]].copy(); beta_score  = fitness[idx_sorted[1]]
    delta_pos = wolves[idx_sorted[2]].copy(); delta_score = fitness[idx_sorted[2]]

    history = [alpha_score]

    for t in range(max_iter):
        a = 2.0 * (1 - t / (max_iter - 1))  # a decreases linearly from 2 to 0

        for i in range(num_wolves):
            X = wolves[i].copy()

            # compute components toward alpha, beta, delta
            r1 = np.random.rand(dim); r2 = np.random.rand(dim)
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = np.abs(C1 * alpha_pos - X)
            X1 = alpha_pos - A1 * D_alpha

            r1 = np.random.rand(dim); r2 = np.random.rand(dim)
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = np.abs(C2 * beta_pos - X)
            X2 = beta_pos - A2 * D_beta

            r1 = np.random.rand(dim); r2 = np.random.rand(dim)
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = np.abs(C3 * delta_pos - X)
            X3 = delta_pos - A3 * D_delta

            # new position
            X_new = (X1 + X2 + X3) / 3.0

            # boundary control
            X_new = np.minimum(np.maximum(X_new, lb), ub)

            # update wolf position and fitness
            wolves[i] = X_new
            fitness[i] = obj_func(X_new)

        # update alpha, beta, delta
        idx_sorted = np.argsort(fitness)
        if fitness[idx_sorted[0]] < alpha_score:
            alpha_pos = wolves[idx_sorted[0]].copy(); alpha_score = fitness[idx_sorted[0]]
        beta_pos = wolves[idx_sorted[1]].copy(); beta_score = fitness[idx_sorted[1]]
        delta_pos = wolves[idx_sorted[2]].copy(); delta_score = fitness[idx_sorted[2]]

        history.append(alpha_score)

        # optional: print progress
        if (t+1) % max(1, max_iter//10) == 0 or t==0:
            print(f"Iter {t+1}/{max_iter}  Best score = {alpha_score:.6e}")

    return alpha_pos, alpha_score, history


# ------------------ Example usage ------------------
if __name__ == "__main__":
    # Define objective: Sphere function (minimum at [0,..,0])
    def sphere(x):
        return np.sum(x**2)

    dim = 2
    lb, ub = -10, 10
    best_pos, best_score, hist = gwo(sphere, lb, ub, dim, num_wolves=20, max_iter=50, seed=0)

    print("\nFinal best position:", best_pos)
    print("Final best score:", best_score)
