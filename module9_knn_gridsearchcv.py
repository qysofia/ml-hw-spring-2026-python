"""
module9_knn_gridsearchcv.py

A mini but fully functional k-Nearest-Neighbors (kNN) classifier with
hyperparameter (k) search implemented via scikit-learn's GridSearchCV.

    Data handling (init / insertion)  -> NumPy
    ML training, search & scoring     -> scikit-learn

Workflow
--------
1. Read N and N (x, y) training pairs            -> TrainS
2. Read M and M (x, y) test pairs                -> TestS
3. Grid-search the best k in [1, 10] using cross-validation on TrainS
4. Report the best k and the accuracy of that model on TestS

Conventions
-----------
x : input feature, a real number
y : class label, a non-negative integer
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# --------------------------------------------------------------------------- #
# Input helpers (robust: keep asking until a valid value is entered)
# --------------------------------------------------------------------------- #
def read_positive_int(prompt):
    """Read a strictly positive integer (> 0)."""
    while True:
        try:
            value = int(input(prompt))
        except (ValueError, EOFError):
            print("  -> Please enter a valid integer.")
            continue
        if value <= 0:
            print("  -> Value must be a positive integer (> 0).")
            continue
        return value


def read_float(prompt):
    """Read a real number (float)."""
    while True:
        try:
            return float(input(prompt))
        except (ValueError, EOFError):
            print("  -> Please enter a valid real number.")


def read_nonneg_int(prompt):
    """Read a non-negative integer (>= 0) used as a class label."""
    while True:
        try:
            value = int(input(prompt))
        except (ValueError, EOFError):
            print("  -> Please enter a valid integer.")
            continue
        if value < 0:
            print("  -> Label must be a non-negative integer (>= 0).")
            continue
        return value


# --------------------------------------------------------------------------- #
# Dataset reading -> NumPy arrays
# --------------------------------------------------------------------------- #
def read_dataset(name):
    """
    Read a labelled set of (x, y) pairs and return NumPy arrays.

    Returns
    -------
    X : np.ndarray, shape (n, 1)  -- features (one real feature per sample)
    y : np.ndarray, shape (n,)    -- non-negative integer labels
    """
    n = read_positive_int(f"Enter the number of {name} pairs: ")

    # Data initialization with NumPy (pre-allocated arrays)
    X = np.empty((n, 1), dtype=float)   # feature column
    y = np.empty(n, dtype=int)          # label vector

    for i in range(n):
        print(f"  {name} pair #{i + 1}:")
        xi = read_float("    x (feature, real number): ")
        yi = read_nonneg_int("    y (label, non-negative integer): ")
        # Data insertion via NumPy indexing
        X[i, 0] = xi
        y[i] = yi

    return X, y


# --------------------------------------------------------------------------- #
# Main program
# --------------------------------------------------------------------------- #
def main():
    print("=== Mini kNN Classifier with GridSearchCV hyperparameter search ===\n")

    print("--- Training set (TrainS) ---")
    X_train, y_train = read_dataset("training")

    print("\n--- Test set (TestS) ---")
    X_test, y_test = read_dataset("test")

    # ------------------------------------------------------------------ #
    # Hyperparameter search over k in [1, 10]
    # ------------------------------------------------------------------ #
    n_train = X_train.shape[0]

    # k can never exceed the number of available training samples.
    max_k = min(10, n_train)
    param_grid = {"n_neighbors": list(range(1, max_k + 1))}

    # GridSearchCV uses StratifiedKFold for classification, so the number of
    # folds cannot exceed the size of the smallest class. Pick a safe value.
    min_class_count = int(np.min(np.bincount(y_train)))

    if n_train < 2 or min_class_count < 2:
        # Not enough data per class for cross-validation:
        # fall back to selecting k by training-set accuracy.
        print("\n[warning] Too few samples per class for cross-validation; "
              "selecting k by training-set accuracy instead.")
        best_k, best_cv_score = param_grid["n_neighbors"][0], -1.0
        for k in param_grid["n_neighbors"]:
            clf = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
            score = clf.score(X_train, y_train)
            if score > best_cv_score:
                best_cv_score, best_k = score, k
        best_model = KNeighborsClassifier(n_neighbors=best_k).fit(X_train, y_train)
    else:
        cv = min(5, min_class_count)   # between 2 and 5 folds
        grid = GridSearchCV(
            estimator=KNeighborsClassifier(),
            param_grid=param_grid,
            cv=cv,
            scoring="accuracy",
        )
        grid.fit(X_train, y_train)
        best_k = grid.best_params_["n_neighbors"]
        best_cv_score = grid.best_score_
        best_model = grid.best_estimator_   # already refit on the full TrainS

    # ------------------------------------------------------------------ #
    # Evaluate the chosen model on the test set
    # ------------------------------------------------------------------ #
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    print("\n=== Results ===")
    print(f"Best k (n_neighbors):                {best_k}")
    print(f"Cross-validation accuracy on TrainS: {best_cv_score:.4f}")
    print(f"Test accuracy on TestS:              {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
