import numpy as np
from sklearn.neighbors import KNeighborsRegressor

# Read N and k
n = int(input("Enter a positive integer N (number of points): "))
k = int(input("Enter a positive integer k (number of neighbors): "))

# Read N points using numpy
points_x = np.zeros(n)
points_y = np.zeros(n)

for i in range(n):
    points_x[i] = float(input(f"Enter x value for point {i+1}: "))
    points_y[i] = float(input(f"Enter y value for point {i+1}: "))

# Read query X
query_x = float(input("Enter X to predict Y: "))

# Check if k <= N
if k > n:
    print(f"Error: k ({k}) must be less than or equal to N ({n}).")
else:
    # Reshape for scikit-learn (requires 2D array)
    X_train = points_x.reshape(-1, 1)
    y_train = points_y

    # Build kNN Regression model using scikit-learn
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)

    # Predict Y for query X
    X_query = np.array([[query_x]])
    predicted_y = model.predict(X_query)

    # Calculate variance of labels using numpy
    variance = np.var(points_y)

    print(f"k-NN Regression result (predicted Y): {predicted_y[0]}")
    print(f"Variance of labels in training dataset: {variance}")
