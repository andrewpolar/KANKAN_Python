# Example of probabilistic model.
# Data: inputs are quantities of virtual dice (10 outcomes). Hidden variables are 
# sums of outcomes of roll. Targets are quadratic forms of symmetric matrices.

import time
import numpy as np

from kankan import KAN, EnsembleBuilder
from quadratic_data import QuadraticDataHolder, calculate_calibration

# Problem parameters.
n = 10
N = 10_000
tree_depth = 8

min_for_targets = -100.0
max_for_targets = 100.0
alpha = 0.2

n_epochs_per_layer = 1
n_initial_epochs = 8
n_targets = 4
n_features = 8

# Generate training data.
data = QuadraticDataHolder(
    n, n_features, n_targets,
    min_for_targets, max_for_targets
)
data.generate_data(N)

features = data.get_features()
targets = data.get_targets()

feature_min = features.min(axis=0).tolist()
feature_max = features.max(axis=0).tolist()
target_min = targets.min(axis=0).tolist()
target_max = targets.max(axis=0).tolist()
# data is ready

# Network architecture and interpolation points.
network = [n_features, 16, n_targets]
grid = [4, 4]

rng = np.random.default_rng()

root_kan = KAN(
    network, grid,
    feature_min, feature_max,
    target_min, target_max,
    alpha,
    int(rng.integers(0, 2**32, dtype=np.uint32))
)

indices = list(range(N))

# Build and train the stochastic KAN ensemble.
builder = EnsembleBuilder(
    features.tolist(),
    targets.tolist(),
    n_epochs_per_layer
)

print("Testing of stochastic system")
print("Features are 8 quantities of dice, 8 hidden random variables are sums of roll outcomes,")
print("4 targets are quadratic forms.")
print("The model is an ensemble of KANs.")
print("Training:")

start = time.perf_counter()

ensemble = builder.build(
    root_kan,
    indices,
    tree_depth,
    2,
    n_initial_epochs
)

print(f"Training time: {time.perf_counter() - start:3.4f} seconds")

# Calibration of the MDDR model.
print("\nCalibration of MDDR model:")
start = time.perf_counter()

for step in range(1, 10):
    probability = 0.10 * step

    within = 0

    for record in range(N):
        # Generate a new, unseen record.
        feature = data.generate_feature()
        hidden = data.generate_hidden(feature)
        target = data.generate_target(hidden)
        target = data.normalize_target(target)

        # Predict using the ensemble.
        sample = ensemble.predict_sample(feature.tolist())

        # Calculate prediction point and radius.
        point, radius = calculate_calibration(sample, probability)

        # Euclidean distance from actual target.
        distance = np.linalg.norm(point - target)

        if distance <= radius:
            within += 1

    actual = within / N
    print(f"Target {probability:3.2f}, actual {actual:3.2f}")

print(f"MDDR calibration time: {time.perf_counter() - start:3.4f} seconds")
