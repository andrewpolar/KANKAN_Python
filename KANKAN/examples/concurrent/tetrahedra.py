
# Example of prediction of areas of faces or random tetrahedra.  
import numpy as np
import time
import copy
from concurrent.futures import ThreadPoolExecutor

from kankan import KAN

def make_random_matrix(rows, cols, min_value, max_value, rng):
    return rng.uniform(min_value, max_value, size=(rows, cols))

def compute_target_matrix(x):
    # Four vertices: A, B, C, D
    a = x[:, 0:3]
    b = x[:, 3:6]
    c = x[:, 6:9]
    d = x[:, 9:12]

    def area(p1, p2, p3):
        cross = np.cross(p2 - p1, p3 - p1)
        return 0.5 * np.linalg.norm(cross, axis=1)

    targets = np.empty((len(x), 4))

    targets[:, 0] = area(a, b, c)
    targets[:, 1] = area(a, b, d)
    targets[:, 2] = area(a, c, d)
    targets[:, 3] = area(b, c, d)

    return targets

def get_column_min_max(matrix):
    return np.min(matrix, axis=0), np.max(matrix, axis=0)
# end data generation

# Dataset and training parameters
n_training_records = 500_000
n_validation_records = 50_000
n_features = 12
n_targets = 4

alpha = 0.2
n_disjoints = 10
n_disjoint_size = 5_000
n_loops = 30

min_for_features = 0.0
max_for_features = 10.0

rng = np.random.default_rng(42)

# Generate datasets
features_training = make_random_matrix(
    n_training_records, n_features,
    min_for_features, max_for_features, rng
)
features_validation = make_random_matrix(
    n_validation_records, n_features,
    min_for_features, max_for_features, rng
)

targets_training = compute_target_matrix(features_training)
targets_validation = compute_target_matrix(features_validation)

# Feature and target ranges
feature_min, feature_max = get_column_min_max(features_training)
target_min, target_max = get_column_min_max(targets_training)

print(
    f"Targets are areas of faces of random tetrahedra, "
    f"{n_training_records} training records"
)
print("Training is concurrent ...")

# Initialize the first model and make independent copies
kan = KAN(
    [n_features, 60, 10, n_targets],
    [2, 12, 22],
    feature_min.tolist(),
    feature_max.tolist(),
    target_min.tolist(),
    target_max.tolist(),
    alpha,
    42
)

kans = [kan] + [
    copy.copy(kan) for _ in range(n_disjoints - 1)
]


def train_disjoint(m, loop):
    start = (
        (loop * n_disjoints + m) * n_disjoint_size
    ) % n_training_records

    for i in range(n_disjoint_size):
        idx = (start + i) % n_training_records
        kans[m].forward(features_training[idx].tolist())
        kans[m].update(targets_training[idx].tolist())


start_time = time.perf_counter()

with ThreadPoolExecutor(max_workers=n_disjoints) as executor:
    for loop in range(n_loops):

        # Train all models concurrently
        list(executor.map(
            lambda m: train_disjoint(m, loop),
            range(n_disjoints)
        ))

        # Average model parameters
        for m in range(1, n_disjoints):
            kans[0].accumulate(kans[m])

        kans[0].scale(1.0 / n_disjoints)

        # Synchronize all models
        for m in range(1, n_disjoints):
            kans[m] = copy.copy(kans[0])

        elapsed = (time.perf_counter() - start_time) * 1000
        print(f"Loop {loop + 1} of {n_loops}, time {elapsed:.0f} ms")

training_time = (time.perf_counter() - start_time) * 1000
print(f"Training is completed in {training_time:.0f} ms")

# Normalize the final model
kans[0].normalize_layers()

# Validate on unseen data
prediction = np.empty((n_validation_records, n_targets))

for record in range(n_validation_records):
    prediction[record] = kans[0].predict(
        features_validation[record].tolist()
    )

pearsons = [
    np.corrcoef(prediction[:, j], targets_validation[:, j])[0, 1]
    for j in range(n_targets)
]

print(
    "Pearsons for unseen data: "
    + ", ".join(f"{p:.6f}" for p in pearsons)
)
