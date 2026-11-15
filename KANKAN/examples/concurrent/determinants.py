# Example of prediction of determinants or random matrices 4 * 4.
import numpy as np
import time
import copy
from concurrent.futures import ThreadPoolExecutor

from kankan import KAN

# data generation
def generate_input(n_records, n_features, min_value, max_value, rng):
    return rng.uniform(
        min_value, max_value, size=(n_records, n_features)
    )

def compute_determinant_targets(x, matrix_size):
    matrices = x.reshape(-1, matrix_size, matrix_size)
    return np.linalg.det(matrices)

def get_column_min_max(matrix):
    return np.min(matrix, axis=0), np.max(matrix, axis=0)

# Dataset parameters
n_training_records = 100_000
n_validation_records = 20_000
matrix_size = 4
n_features = matrix_size * matrix_size
n_targets = 1

alpha = 0.15
n_disjoints = 3
n_disjoint_size = 38_000
n_loops = 12

min_for_features = 0.0
max_for_features = 10.0

rng = np.random.default_rng(42)

# Generate training and validation datasets
features_training = generate_input(
    n_training_records, n_features,
    min_for_features, max_for_features, rng
)
features_validation = generate_input(
    n_validation_records, n_features,
    min_for_features, max_for_features, rng
)

targets_training = compute_determinant_targets(
    features_training, matrix_size
)
targets_validation = compute_determinant_targets(
    features_validation, matrix_size
)

# Feature and target ranges
feature_min, feature_max = get_column_min_max(features_training)
target_min = [float(np.min(targets_training))]
target_max = [float(np.max(targets_training))]

print(
    f"Targets are determinants of {matrix_size} * {matrix_size} "
    f"random matrices, {n_training_records} training records"
)
print("Training is concurrent ...")

# Create the initial model and its copies
kan = KAN(
    [n_features, 70, n_targets],
    [4, 16],
    feature_min.tolist(),
    feature_max.tolist(),
    target_min,
    target_max,
    alpha,
    42
)

kans = [kan] + [copy.copy(kan) for _ in range(n_disjoints - 1)]


def train_disjoint(m, loop):
    start = (loop * n_disjoints + m) * n_disjoint_size % n_training_records

    for i in range(n_disjoint_size):
        idx = (start + i) % n_training_records
        kans[m].forward(features_training[idx].tolist())
        kans[m].update(float(targets_training[idx]))


start_time = time.perf_counter()

with ThreadPoolExecutor(max_workers=n_disjoints) as executor:
    for loop in range(n_loops):
        # Train independent models concurrently
        list(executor.map(
            lambda m: train_disjoint(m, loop),
            range(n_disjoints)
        ))

        # Average model parameters
        for m in range(1, n_disjoints):
            kans[0].accumulate(kans[m])

        kans[0].scale(1.0 / n_disjoints)

        # Synchronize the models
        for m in range(1, n_disjoints):
            kans[m] = copy.copy(kans[0])

        elapsed = (time.perf_counter() - start_time) * 1000
        print(f"Loop {loop + 1} of {n_loops}, time {elapsed:.0f} ms")

training_time = (time.perf_counter() - start_time) * 1000
print(f"Training is completed in {training_time:.0f} ms")

# Normalize the final model
kans[0].normalize_layers()

# Validate on unseen data
prediction = np.empty(n_validation_records)

for record in range(n_validation_records):
    prediction[record] = kans[0].predict(
        features_validation[record].tolist()
    )[0]

pearson = np.corrcoef(prediction, targets_validation)[0, 1]
print(f"Pearson for unseen data: {pearson:.6f}")