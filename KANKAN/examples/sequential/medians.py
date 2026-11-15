# example of training to predict lengths of medians of random triangles. 

import numpy as np
from math import sqrt
import time
from kankan import KAN

# building data
def median1(x1, y1, x2, y2, x3, y3):
    return sqrt((x1 - (x2 + x3) / 2)**2 +
                (y1 - (y2 + y3) / 2)**2)

def median2(x1, y1, x2, y2, x3, y3):
    return sqrt((x2 - (x1 + x3) / 2)**2 +
                (y2 - (y1 + y3) / 2)**2)

def median3(x1, y1, x2, y2, x3, y3):
    return sqrt((x3 - (x2 + x1) / 2)**2 +
                (y3 - (y2 + y1) / 2)**2)

def generate_inputs_medians(n_records, n_features, min_value, max_value, rng):
    return rng.uniform(min_value, max_value, size=(n_records, n_features))

def generate_inputs_medians2(
    n_records, n_features, min_x, max_x, min_y, max_y, rng
):
    x = np.empty((n_records, n_features))
    x[:, 0::2] = rng.uniform(min_x, max_x, size=(n_records, len(range(0, n_features, 2))))
    x[:, 1::2] = rng.uniform(min_y, max_y, size=(n_records, len(range(1, n_features, 2))))
    return x

def compute_targets_medians(x):
    x1, y1 = x[:, 0], x[:, 1]
    x2, y2 = x[:, 2], x[:, 3]
    x3, y3 = x[:, 4], x[:, 5]

    y = np.empty((len(x), 3))

    y[:, 0] = np.sqrt(
        (x1 - (x2 + x3) / 2)**2 +
        (y1 - (y2 + y3) / 2)**2
    )

    y[:, 1] = np.sqrt(
        (x2 - (x1 + x3) / 2)**2 +
        (y2 - (y1 + y3) / 2)**2
    )

    y[:, 2] = np.sqrt(
        (x3 - (x1 + x2) / 2)**2 +
        (y3 - (y1 + y2) / 2)**2
    )

    return y
# end data building

# getting min/max values
def get_column_min_max(matrix):
    if matrix.size == 0 or matrix.shape[1] == 0:
        return np.array([]), np.array([])
    return np.min(matrix, axis=0), np.max(matrix, axis=0)

# Dataset parameters
n_training_records = 10_000
n_validation_records = 2_000
n_features = 6
n_targets = 3
min_for_features = 0.0
max_for_features = 10.0
alpha = 0.2
n_epochs = 16

rng = np.random.default_rng(42)

# Generate datasets
features_training = generate_inputs_medians(
    n_training_records, n_features,
    min_for_features, max_for_features, rng
)
features_validation = generate_inputs_medians(
    n_validation_records, n_features,
    min_for_features, max_for_features, rng
)

targets_training = compute_targets_medians(features_training)
targets_validation = compute_targets_medians(features_validation)

# Feature and target ranges
feature_min, feature_max = get_column_min_max(features_training)
target_min, target_max = get_column_min_max(targets_training)

print(f"Targets are medians of random triangles, {n_training_records} training records")
print("Training is sequential ...")

# Create KAN
kan = KAN(
    [n_features, 20, 10, 4, n_targets],
    [2, 12, 12, 22],
    feature_min.tolist(),
    feature_max.tolist(),
    target_min.tolist(),
    target_max.tolist(),
    alpha,
    42
)

# Sequential training
start = time.perf_counter()

for epoch in range(n_epochs):
    for record in range(n_training_records):
        kan.forward(features_training[record].tolist())
        kan.update(targets_training[record].tolist())

    print(epoch + 1, end=" ", flush=True)

print()

training_time = (time.perf_counter() - start) * 1000
print(f"Training is completed in {training_time:.0f} ms")

# Normalize layers
kan.normalize_layers()

# Validation
prediction = np.empty((n_validation_records, n_targets))

for record in range(n_validation_records):
    prediction[record] = kan.predict(
        features_validation[record].tolist()
    )

pearsons = [
    np.corrcoef(prediction[:, j], targets_validation[:, j])[0, 1]
    for j in range(n_targets)
]

print(
    "Pearsons for unseen data: "
    + " ".join(f"{p:.4f}" for p in pearsons)
)
