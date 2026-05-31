import numpy as np
from sklearn.metrics import precision_score, recall_score

# Read N
n = int(input("Enter a positive integer N (number of points): "))

# Initialize numpy arrays
true_labels = np.zeros(n, dtype=int)
pred_labels = np.zeros(n, dtype=int)

# Read N points
for i in range(n):
    x = int(input(f"Enter ground truth (X) for point {i+1} (0 or 1): "))
    y = int(input(f"Enter predicted label (Y) for point {i+1} (0 or 1): "))
    true_labels[i] = x
    pred_labels[i] = y

# Compute Precision and Recall using scikit-learn
precision = precision_score(true_labels, pred_labels)
recall = recall_score(true_labels, pred_labels)

print(f"\nPrecision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
