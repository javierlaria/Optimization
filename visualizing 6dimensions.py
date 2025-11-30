import numpy as np
import matplotlib.pyplot as plt

# Given matrices
A = np.array([
    [1.,   2.,   0.,   1.,   0.5, 4.5],
    [2.,   1.,  10.,   2.,   2.5, 1.6],
    [2.,   0.,   5.3,  2.,   0.,   0. ],
    [0.,   0.,   4.3,  6.4,  0.,   2. ]
])

b = np.array([4., 5., 2.5, 2.6])

# -------------------------
# Settings
# -------------------------

# Choose two variables to visualize, e.g. x0 and x1
i, j = 0, 1

# Fix all other variables to zero for projection
x_fixed = np.zeros(6)

# Grid range
xmin, xmax = -2, 8
N = 400

# -------------------------
# Create grid for x_i and x_j
# -------------------------
x_vals = np.linspace(xmin, xmax, N)
y_vals = np.linspace(xmin, xmax, N)
X, Y = np.meshgrid(x_vals, y_vals)

# Evaluate all inequalities on the grid
feasible = np.ones_like(X, dtype=bool)

for k in range(A.shape[0]):
    # Compute A_k x = A_k[i]*X + A_k[j]*Y + constant_term
    const = np.dot(A[k], x_fixed) - A[k, i]*0 - A[k, j]*0  # other fixed vars
    inequality = A[k, i] * X + A[k, j] * Y + const >= b[k]

    feasible &= inequality  # intersection of halfspaces

    # Visualize each inequality boundary
    plt.contour(X, Y, A[k, i]*X + A[k, j]*Y + const - b[k],
                levels=[0], colors='k', linewidths=1)

# Shade feasible region
plt.imshow(feasible.astype(int),
           extent=[xmin, xmax, xmin, xmax],
           origin='lower',
           cmap='Greens',
           alpha=0.4)

plt.title(f"Projection of A x ≥ b onto (x{i}, x{j}) plane")
plt.xlabel(f"x{i}")
plt.ylabel(f"x{j}")
plt.grid(True)
plt.show()
