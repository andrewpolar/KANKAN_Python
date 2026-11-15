# Example of training to predict areas of random triangles. 
import numpy as np
import time
from kankan import KAN

# data generation
def make_random_matrix_for_triangles(rows, cols, min_value, max_value, rng):
    return rng.uniform(min_value, max_value, size=(rows, cols))

def area_of_triangle(x1, y1, x2, y2, x3, y3):
    return 0.5 * abs(
        x1 * (y2 - y3)
        + x2 * (y3 - y1)
        + x3 * (y1 - y2)
    )

def compute_areas_of_triangles(matrix):
    x1, y1 = matrix[:, 0], matrix[:, 1]
    x2, y2 = matrix[:, 2], matrix[:, 3]
    x3, y3 = matrix[:, 4], matrix[:, 5]

    return 0.5 * np.abs(
        x1 * (y2 - y3)
        + x2 * (y3 - y1)
        + x3 * (y1 - y2)
    )

def get_column_min_max(matrix):
    if matrix.size == 0 or matrix.shape[1] == 0:
        return np.array([]), np.array([])

    return np.min(matrix, axis=0), np.max(matrix, axis=0)
# end of data generation

# Dataset parameters
n_targets = 1
n_features = 6
n_training_records = 10_000
n_validation_records = 2_000
min_for_features = 0.0
max_for_features = 10.0
alpha = 0.05
n_epochs = 16

# Random number generator
rng = np.random.default_rng(42)

# Generate datasets
features_training = make_random_matrix_for_triangles(
    n_training_records, n_features,
    min_for_features, max_for_features, rng
)
features_validation = make_random_matrix_for_triangles(
    n_validation_records, n_features,
    min_for_features, max_for_features, rng
)

targets_training = compute_areas_of_triangles(features_training)
targets_validation = compute_areas_of_triangles(features_validation)

# Calculate feature and target ranges
feature_min, feature_max = get_column_min_max(features_training)
target_min = np.array([targets_training.min()])
target_max = np.array([targets_training.max()])

print(f"Targets are areas of random triangles, {n_training_records} training records")
print("Training is sequential ...")

# Create KAN
kan = KAN(
    [n_features, 50, 10, 4, n_targets],
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
        kan.update(float(targets_training[record]))

    print(epoch + 1, end=" ", flush=True)

print()

training_time = (time.perf_counter() - start) * 1000
print(f"Training is completed in {training_time:.0f} ms")

# Normalize layers
kan.normalize_layers()

# Validation
prediction = np.empty(n_validation_records)

for record in range(n_validation_records):
    prediction[record] = kan.predict(
        features_validation[record].tolist()
    )[0]

pearson = np.corrcoef(prediction, targets_validation)[0, 1]
print(f"Pearson for unseen data: {pearson:.4f}")
