
import numpy as np


class QuadraticDataHolder:
    def __init__(self, n_size, n_features, n_targets,
                 min_target, max_target, seed=None):

        if n_size < 2:
            raise ValueError("n_size must be at least 2.")
        if n_features < 1:
            raise ValueError("n_features must be at least 1.")
        if n_targets < 1:
            raise ValueError("n_targets must be at least 1.")
        if max_target <= min_target:
            raise ValueError("max_target must exceed min_target.")

        self.n_size = n_size
        self.n_features = n_features
        self.n_targets = n_targets

        self.rng = np.random.default_rng(seed)

        self.normal_min_target = min_target
        self.normal_max_target = max_target
        self.is_normalized = False

        self.target_raw_min = None
        self.target_raw_max = None

        # Generate one symmetric quadratic matrix per target.
        self.mapping_matrices = []

        for _ in range(n_targets):
            values = self.rng.integers(
                0, 200, size=(n_features, n_features)
            )
            upper = np.triu(values)
            matrix = (upper + np.triu(upper, 1).T - 100) / 100.0
            self.mapping_matrices.append(matrix)

        self.features = None
        self.targets = None

    def generate_feature(self):
        return self.rng.integers(
            1, self.n_size + 1, size=self.n_features
        ).astype(float)

    def generate_hidden(self, feature):
        if len(feature) != self.n_features:
            raise ValueError("Feature vector has incorrect size.")

        dice_counts = np.asarray(feature + 0.1, dtype=int)
        return np.array([
            self.rng.integers(1, self.n_size + 1, size=count).sum()
            for count in dice_counts
        ], dtype=float)

    def generate_target(self, hidden):
        if len(hidden) != self.n_features:
            raise ValueError("Hidden vector has incorrect size.")

        return np.array([
            hidden @ matrix @ hidden
            for matrix in self.mapping_matrices
        ])

    def find_raw_limits(self):
        if self.is_normalized:
            return

        if len(self.targets) == 0:
            raise ValueError("No targets to normalize.")

        self.target_raw_min = np.min(self.targets, axis=0)
        self.target_raw_max = np.max(self.targets, axis=0)
        self.is_normalized = True

    def normalize_target(self, target):
        if not self.is_normalized:
            raise RuntimeError(
                "Target normalization limits are not initialized."
            )

        result = np.empty(self.n_targets)

        ranges = self.target_raw_max - self.target_raw_min
        mask = ranges == 0.0

        result[mask] = self.normal_min_target
        result[~mask] = (
            (target[~mask] - self.target_raw_min[~mask])
            / ranges[~mask]
            * (self.normal_max_target - self.normal_min_target)
            + self.normal_min_target
        )

        return result

    def generate_data(self, n_records):
        if n_records < 1:
            raise ValueError("n_records must be at least 1.")

        self.features = np.array([
            self.generate_feature() for _ in range(n_records)
        ])

        hidden = np.array([
            self.generate_hidden(feature) for feature in self.features
        ])

        self.targets = np.array([
            self.generate_target(record) for record in hidden
        ])

        self.find_raw_limits()

        self.targets = np.array([
            self.normalize_target(target) for target in self.targets
        ])

    def get_features(self):
        return self.features

    def get_targets(self):
        return self.targets

    def get_mapping_matrices(self):
        return self.mapping_matrices



def calculate_calibration(sample, probability):
    sample = np.asarray(sample, dtype=float)

    if sample.ndim != 2 or sample.shape[0] == 0:
        raise ValueError("Sample must be a non-empty 2D array.")

    if not 0.0 <= probability <= 1.0:
        raise ValueError("Probability must be between 0 and 1.")

    # Component-wise median.
    central_point = np.median(sample, axis=0)

    # Euclidean distances from the median point.
    distances = np.linalg.norm(sample - central_point, axis=1)

    # Radius covering the requested fraction of points.
    distances.sort()
    count = int(np.ceil(probability * len(distances)))

    radius = 0.0 if count == 0 else distances[count - 1]

    return central_point, radius
