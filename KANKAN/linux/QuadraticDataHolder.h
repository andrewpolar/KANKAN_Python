
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

class QuadraticDataHolder {
private:
    int nSize;
    int nFeatures;
    int nTargets;

    std::mt19937_64 rng;
    std::vector<double> targetRawMin;
    std::vector<double> targetRawMax;
    double normalMinTarget;
    double normalMaxTarget;
    bool isNormalized;

    std::vector<std::vector<double>> features;
    std::vector<std::vector<double>> targets;

    // One quadratic matrix Q per target.
    // target[j] = x^T Q[j] x
    std::vector<std::vector<std::vector<double>>> MappingMatrices;

    std::vector<std::vector<std::vector<double>>> GenerateQuadraticMatrices() {
        std::vector<std::vector<std::vector<double>>> matrices;

        for (int target = 0; target < nTargets; ++target)
            matrices.push_back(GenerateQuadraticMatrix());

        return matrices;
    }

    std::vector<std::vector<double>> GenerateQuadraticMatrix() {
        std::vector<std::vector<double>> Q(nFeatures, std::vector<double>(nFeatures, 0.0));

        std::uniform_int_distribution<int> distribution(0, 199);
        for (int i = 0; i < nFeatures; ++i) {
            for (int j = i; j < nFeatures; ++j) {
                double value = (distribution(rng) - 100.0) / 100.0;
                Q[i][j] = value;
                Q[j][i] = value;
            }
        }
        return Q;
    }

    void FindRawLimits() {
        if (isNormalized) return;

        if (targets.empty())
            throw std::runtime_error("No targets to normalize.");

        targetRawMin = targets[0];
        targetRawMax = targets[0];

        for (const auto& target : targets) {
            for (int i = 0; i < nTargets; ++i) {
                targetRawMin[i] = std::min(targetRawMin[i], target[i]);
                targetRawMax[i] = std::max(targetRawMax[i], target[i]);
            }
        }
        isNormalized = true;
    }

    void NormalizeAllTargets() {
        for (auto& target : targets)
            NormalizeTarget(target);
    }

public:
    QuadraticDataHolder(int nSize, int nFeatures, int nTargets, double minTarget,
        double maxTarget, std::uint64_t seed = std::random_device{}()): 
        nSize(nSize),
        nFeatures(nFeatures),
        nTargets(nTargets),
        rng(seed),
        normalMinTarget(minTarget),
        normalMaxTarget(maxTarget),
        isNormalized(false) {

        if (nSize < 2)
            throw std::invalid_argument(
                "nSize must be at least 2.");

        if (nFeatures < 1)
            throw std::invalid_argument(
                "nFeatures must be at least 1.");

        if (nTargets < 1)
            throw std::invalid_argument(
                "nTargets must be at least 1.");

        if (maxTarget <= minTarget)
            throw std::invalid_argument(
                "maxTarget must exceed minTarget.");

        targetRawMin.resize(nTargets);
        targetRawMax.resize(nTargets);
        MappingMatrices = GenerateQuadraticMatrices();
    }

    // Generate a new dataset.
    // Q matrices and established target normalization limits
    // are retained across calls.
    void GenerateData(int nRecords) {
        if (nRecords < 1)
            throw std::invalid_argument(
                "nRecords must be at least 1.");

        features.clear();
        targets.clear();

        features.reserve(nRecords);
        targets.reserve(nRecords);

        for (int record = 0; record < nRecords; ++record) {
            auto feature = GenerateFeature();
            auto hidden = GenerateHidden(feature);
            auto target = GenerateTarget(hidden);

            features.push_back(std::move(feature));
            targets.push_back(std::move(target));
        }

        FindRawLimits();
        NormalizeAllTargets();
    }

    // Returns dice counts in [1, nSize], matching the
    // actual behavior of the supplied C# implementation.
    std::vector<double> GenerateFeature() {
        std::vector<double> feature(nFeatures);
        std::uniform_int_distribution<int> distribution(1, nSize);
        for (double& value : feature)
            value = distribution(rng);
        return feature;
    }

    std::vector<double> GenerateHidden(const std::vector<double>& feature) {
        if (static_cast<int>(feature.size()) != nFeatures)
            throw std::invalid_argument(
                "Feature vector has incorrect size.");

        std::vector<double> hidden(nFeatures);

        std::uniform_int_distribution<int> distribution(1, nSize);

        for (int i = 0; i < nFeatures; ++i) {
            int numberOfDice = static_cast<int>(feature[i] + 0.1);
            double sum = 0.0;
            for (int dice = 0; dice < numberOfDice; ++dice)
                sum += distribution(rng);

            hidden[i] = sum;
        }
        return hidden;
    }

    std::vector<double> GenerateTarget(const std::vector<double>& hidden) {
        if (static_cast<int>(hidden.size()) != nFeatures)
            throw std::invalid_argument(
                "Hidden vector has incorrect size.");

        std::vector<double> target(nTargets, 0.0);

        for (int j = 0; j < nTargets; ++j) {
            const auto& Q = MappingMatrices[j];
            double value = 0.0;

            for (int row = 0; row < nFeatures; ++row) {
                for (int col = 0; col < nFeatures; ++col) {
                    value += hidden[row] * Q[row][col] * hidden[col];
                }
            }
            target[j] = value;
        }
        return target;
    }

    void NormalizeTarget(std::vector<double>& target) {
        if (!isNormalized)
            throw std::runtime_error(
                "Target normalization limits are not initialized.");

        if (static_cast<int>(target.size()) != nTargets)
            throw std::invalid_argument(
                "Target vector has incorrect size.");

        for (int j = 0; j < nTargets; ++j) {
            double range = targetRawMax[j] - targetRawMin[j];

            if (range == 0.0) {
                target[j] = normalMinTarget;
            }
            else {
                target[j] = (target[j] - targetRawMin[j]) / range
                    * (normalMaxTarget - normalMinTarget)
                    + normalMinTarget;
            }
        }
    }

    const std::vector<std::vector<double>>& GetFeatures() const {
        return features;
    }

    const std::vector<std::vector<double>>& GetTargets() const {
        return targets;
    }

    const std::vector<std::vector<std::vector<double>>>& GetMappingMatrices() const {
        return MappingMatrices;
    }
};
