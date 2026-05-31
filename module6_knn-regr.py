import numpy as np

# Read N and k
n = int(input("Enter a positive integer N (number of points): "))
k = int(input("Enter a positive integer k (number of neighbors): "))

# Read N points
points_x = []
points_y = []

for i in range(n):
    x = float(input(f"Enter x value for point {i+1}: "))
    y = float(input(f"Enter y value for point {i+1}: "))
    points_x.append(x)
    points_y.append(y)

# Convert to numpy arrays
points_x = np.array(points_x)
points_y = np.array(points_y)

# Read query X
query_x = float(input("Enter X to predict Y: "))

# Check if k <= N
if k > n:
    print(f"Error: k ({k}) must be less than or equal to N ({n}).")
else:
    # Calculate distances using numpy
    distances = np.abs(points_x - query_x)

    # Get indices of k nearest neighbors
    k_nearest_indices = np.argsort(distances)[:k]

    # Get Y values of k nearest neighbors
    k_nearest_y = points_y[k_nearest_indices]

    # Calculate mean (kNN regression result)
    result = np.mean(k_nearest_y)

    print(f"k-NN Regression result (predicted Y): {result}")
