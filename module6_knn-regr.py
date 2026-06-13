import numpy as np

# Read N and k
n = int(input("Enter a positive integer N (number of points): "))
k = int(input("Enter a positive integer k (number of neighbors): "))

# Data initialization using NumPy (no Python list / append)
points_x = np.empty(n, dtype=float)
points_y = np.empty(n, dtype=float)

# Data insertion using NumPy indexing
for i in range(n):
    points_x[i] = float(input(f"Enter x value for point {i+1}: "))
    points_y[i] = float(input(f"Enter y value for point {i+1}: "))

# Read query X
query_x = float(input("Enter X to predict Y: "))

# Check if k <= N
if k > n:
    print(f"Error: k ({k}) must be less than or equal to N ({n}).")
else:
    # Data calculation using NumPy
    distances = np.abs(points_x - query_x)
    k_nearest_indices = np.argsort(distances)[:k]
    result = np.mean(points_y[k_nearest_indices])
    print(f"k-NN Regression result (predicted Y): {result}")
