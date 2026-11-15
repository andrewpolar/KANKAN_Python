#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

class Splitter {
public:
    using Data = std::vector<std::vector<double>>;

    struct SplitResult {
        std::vector<int> Above;
        std::vector<int> BelowOrEqual;
    };

private:
    const Data& Items;
    std::vector<int> Indices;

    static std::mt19937_64& RandomEngine() {
        static thread_local std::mt19937_64 engine{
            std::random_device{}()
        };
        return engine;
    }

    static double ComputeInnerProduct(const std::vector<double>& a, const std::vector<double>& b) {
        double sum = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i)
            sum += a[i] * b[i];
        return sum;
    }

    static void Normalize(std::vector<double>& v) {
        double norm = 0.0;

        for (double x : v) norm += x * x;
        norm = std::sqrt(norm);
        if (norm > 0.0) {
            for (double& x : v) x /= norm;
        }
    }

    static double CalculateMedian(const std::vector<double>& sortedValues) {
        const std::size_t count = sortedValues.size();

        if (count % 2 == 1) return sortedValues[count / 2];
        return (sortedValues[count / 2 - 1] + sortedValues[count / 2]) / 2.0;
    }

    static std::vector<double> GenerateRandomUnitVector(int size) {
        std::vector<double> v(size, 0.0);

        if (size == 1) {
            v[0] = 1.0;
            return v;
        }

        std::normal_distribution<double> normal(0.0, 1.0);
        double normSum = 0.0;
        for (double& x : v) {
            x = normal(RandomEngine());
            normSum += x * x;
        }
        double magnitude = std::sqrt(normSum);
        if (magnitude > 1e-9) {
            for (double& x : v) x /= magnitude;
        }
        return v;
    }

    static std::vector<double> GenerateNormalUnitVector(int vectorSize, int layer) {
        std::vector<double> v(vectorSize, 0.0);
        v[layer % vectorSize] = 1.0;
        return v;
    }

    static std::vector<double> GeneratePrincipalDirectionUnitVector(const Data& items, const std::vector<int>& indices)
    {
        if (indices.size() < 2)
            throw std::invalid_argument(
                "At least two vectors are required.");

        const std::size_t dimension = items.at(indices[0]).size();

        if (dimension == 0)
            throw std::invalid_argument(
                "Vectors cannot be empty.");

        std::vector<double> mean(dimension, 0.0);

        // Calculate mean.
        for (int index : indices) {
            const auto& x = items.at(index);

            if (x.size() != dimension)
                throw std::invalid_argument(
                    "All vectors must have the same dimension.");

            for (std::size_t j = 0; j < dimension; ++j)
                mean[j] += x[j];
        }

        for (double& x : mean) x /= indices.size();

        // Calculate covariance matrix.
        std::vector<std::vector<double>> cov(dimension, std::vector<double>(dimension, 0.0));

        for (int index : indices) {
            const auto& x = items.at(index);
            for (std::size_t i = 0; i < dimension; ++i) {
                const double di = x[i] - mean[i];
                for (std::size_t j = i; j < dimension; ++j)
                    cov[i][j] += di * (x[j] - mean[j]);
            }
        }

        const double denominator = (double)(indices.size() - 1);

        for (std::size_t i = 0; i < dimension; ++i) {
            for (std::size_t j = i; j < dimension; ++j) {
                cov[i][j] /= denominator;
                cov[j][i] = cov[i][j];
            }
        }

        // Power iteration.
        std::vector<double> v(dimension);

        for (std::size_t i = 0; i < dimension; ++i)
            v[i] = 1.0 / std::sqrt(static_cast<double>(dimension));

        for (std::size_t i = 0; i < dimension; ++i)
            v[i] += (i + 1.0) * 1e-4;

        Normalize(v);

        std::vector<double> next(dimension);

        for (int iteration = 0; iteration < 100; ++iteration) {
            std::fill(next.begin(), next.end(), 0.0);

            for (std::size_t i = 0; i < dimension; ++i) {
                for (std::size_t j = 0; j < dimension; ++j)
                    next[i] += cov[i][j] * v[j];
            }

            double norm = 0.0;
            for (double x : next) norm += x * x;

            norm = std::sqrt(norm);

            if (norm < 1e-15) {
                std::fill(v.begin(), v.end(), 0.0);
                v[0] = 1.0;
                return v;
            }

            for (double& x : next) x /= norm;

            double alignment = 0.0;

            for (std::size_t i = 0; i < dimension; ++i)
                alignment += next[i] * v[i];

            if (alignment < 0.0) {
                for (double& x : next) x = -x;
                alignment = -alignment;
            }

            v = next;
            if (1.0 - alignment < 1e-12)
                break;
        }
        return v;
    }

public:
    Splitter(const Data& items, std::vector<int> indices): 
        Items(items), Indices(std::move(indices)) {
        if (Indices.empty())
            throw std::invalid_argument(
                "Indices cannot be empty.");
    }

    const std::vector<int>& GetIndices() const {
        return Indices;
    }

    SplitResult Split(int mode, int layer) const {
        if (Indices.size() <= 1)
            throw std::runtime_error(
                "Cannot split a node containing 1 or 0 elements.");

        const int vectorSize = static_cast<int>(Items.at(Indices[0]).size());

        if (vectorSize <= 0)
            throw std::invalid_argument(
                "Vectors cannot be empty.");

        std::vector<double> unitVector;

        if (mode == 0)
            unitVector = GenerateNormalUnitVector(vectorSize, layer);
        else if (mode == 1)
            unitVector = GenerateRandomUnitVector(vectorSize);
        else
            unitVector = GeneratePrincipalDirectionUnitVector(Items, Indices);

        std::vector<std::pair<int, double>> products;
        products.reserve(Indices.size());

        for (int index : Indices) {
            double product = ComputeInnerProduct(Items.at(index), unitVector);
            products.emplace_back(index, product);
        }

        std::sort(products.begin(), products.end(), [](const auto& a, const auto& b) {
            return a.second < b.second;
        });

        std::vector<double> sortedValues;
        sortedValues.reserve(products.size());

        for (const auto& p : products)
            sortedValues.push_back(p.second);

        const double median = CalculateMedian(sortedValues);

        SplitResult result;

        for (const auto& p : products) {
            if (p.second >= median)
                result.Above.push_back(p.first);
            else
                result.BelowOrEqual.push_back(p.first);
        }

        // Fallback split.
        if (result.Above.empty() || result.BelowOrEqual.empty()) {
            const std::size_t midpoint = products.size() / 2;

            result.Above.clear();
            result.BelowOrEqual.clear();

            for (std::size_t i = 0; i < midpoint; ++i)
                result.BelowOrEqual.push_back(products[i].first);

            for (std::size_t i = midpoint; i < products.size(); ++i)
                result.Above.push_back(products[i].first);
        }
        return result;
    }
};

class KANLeaf {
public:
    KAN Model;
    std::vector<int> Indices;

    KANLeaf(const KAN& model, const std::vector<int>& indices): Model(model), Indices(indices)
    {
    }
};

class KANEnsemble {
private:
    std::vector<KANLeaf> Leaves;

public:
    explicit KANEnsemble(std::vector<KANLeaf> leaves) : Leaves(std::move(leaves))
    {
    }

    const std::vector<KANLeaf>& GetLeaves() const {
        return Leaves;
    }

    std::vector<std::vector<double>> PredictSample(const std::vector<double>& features) {
        if (Leaves.empty())
            throw std::runtime_error(
                "The ensemble contains no models.");

        std::vector<std::vector<double>> sample;
        sample.reserve(Leaves.size());

        for (auto& leaf : Leaves) {
            leaf.Model.Predict(features);
            const auto& prediction = leaf.Model.GetPrediction();
            sample.emplace_back(prediction.begin(), prediction.end());
        }
        return sample;
    }
};

class EnsembleBuilder {
private:
    const std::vector<std::vector<double>>& Features;
    const std::vector<std::vector<double>>& Targets;

    int EpochsPerLayer;

    void TrainModel(KAN& model, const std::vector<int>& indices, int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (int index : indices) {
                model.Forward(Features.at(index));
                model.Update(Targets.at(index));
            }
        }
    }

    std::vector<std::vector<double>> ComputeResiduals(KAN& model, const std::vector<int>& indices) {
        std::vector<std::vector<double>> residuals;
        residuals.reserve(indices.size());

        for (int index : indices) {
            model.Predict(Features.at(index));

            const auto& prediction = model.GetPrediction();
            const auto& target = Targets.at(index);

            std::vector<double> residual(prediction.size());

            for (std::size_t j = 0; j < prediction.size(); ++j)
                residual[j] = target[j] - prediction[j];

            residuals.push_back(std::move(residual));
        }
        return residuals;
    }

public:
    EnsembleBuilder(
        const std::vector<std::vector<double>>& features,
        const std::vector<std::vector<double>>& targets,
        int epochsPerLayer = 1): 
        Features(features),
        Targets(targets),
        EpochsPerLayer(epochsPerLayer) {
        if (Features.size() != Targets.size())
            throw std::invalid_argument(
                "Features and targets must have the same number of records.");

        if (EpochsPerLayer < 1)
            throw std::invalid_argument(
                "epochsPerLayer must be positive.");
    }

    KANEnsemble Build(const KAN& rootModel, const std::vector<int>& rootIndices,
        int numberOfSplittingLayers, int mode, int initialTrainingEpochs = 8) {
        if (rootIndices.empty())
            throw std::invalid_argument(
                "Root node contains no records.");

        // First model needs longer training.
        KAN root = rootModel;

        TrainModel(root, rootIndices, initialTrainingEpochs);

        std::vector<KANLeaf> currentLayer;
        currentLayer.emplace_back(root, rootIndices);

        for (int layer = 0; layer < numberOfSplittingLayers; ++layer) {
            std::vector<KANLeaf> nextLayer;
            nextLayer.reserve(currentLayer.size() * 2);

            for (const KANLeaf& parent : currentLayer) {
                // Residuals are stored in the same order as parent.Indices.
                auto residuals = ComputeResiduals(const_cast<KAN&>(parent.Model), parent.Indices);

                // These are local positions in the residuals array.
                std::vector<int> localIndices(residuals.size());

                std::iota(localIndices.begin(), localIndices.end(), 0);

                Splitter splitter(residuals, localIndices);
                auto split = splitter.Split(mode, layer);

                // Convert local positions back to dataset indices.
                std::vector<int> aboveGlobalIndices;
                std::vector<int> belowGlobalIndices;

                for (int local : split.Above)
                    aboveGlobalIndices.push_back(parent.Indices.at(local));

                for (int local : split.BelowOrEqual)
                    belowGlobalIndices.push_back(parent.Indices.at(local));

                // Copy the parent model into both children.
                KAN aboveModel = parent.Model;
                KAN belowModel = parent.Model;

                TrainModel(aboveModel, aboveGlobalIndices, EpochsPerLayer);
                TrainModel(belowModel, belowGlobalIndices, EpochsPerLayer);

                nextLayer.emplace_back(aboveModel, aboveGlobalIndices);
                nextLayer.emplace_back(belowModel, belowGlobalIndices);
            }

            currentLayer = std::move(nextLayer);
            printf("Layer %d: %zu KAN models\n", layer + 1, currentLayer.size());
        }

        // Final training.
        for (KANLeaf& leaf : currentLayer) {
            TrainModel(leaf.Model, leaf.Indices, EpochsPerLayer);
        }

        printf("Final training completed for %zu KAN models.\n", currentLayer.size());
        return KANEnsemble(std::move(currentLayer));
    }
};
