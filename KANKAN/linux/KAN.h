
#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <stdexcept>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

class KAN {
public:

    KAN(std::vector<int> Network, std::vector<int> Grid,
        const std::vector<double>& featureMin,
        const std::vector<double>& featureMax,
        const std::vector<double>& targetMin,
        const std::vector<double>& targetMax,
        double learningRate, std::mt19937& RND) {

        if (Network.size() != Grid.size() + 1 || Network.empty()) {
            throw std::invalid_argument(
                "Network must contain exactly one more element than Grid.");
        }

        if (featureMin.size() != (size_t)Network[0] ||
            featureMax.size() != (size_t)Network[0] ||
            targetMin.size() != (size_t)Network.back() ||
            targetMax.size() != (size_t)Network.back()) {
            throw std::invalid_argument("Range vector size mismatch.");
        }

        // Store original ranges
        this->featureMin = featureMin;
        this->featureMax = featureMax;
        this->targetMin = targetMin;
        this->targetMax = targetMax;

        // Check ranges
        for (size_t i = 0; i < featureMin.size(); ++i) {
            if (featureMax[i] <= featureMin[i])
                throw std::invalid_argument("Invalid feature range.");
        }

        for (size_t i = 0; i < targetMin.size(); ++i) {
            if (targetMax[i] <= targetMin[i])
                throw std::invalid_argument("Invalid target range.");
        }

        // Allocate reusable normalization buffers
        normalizedFeatures.resize(featureMin.size());
        normalizedTargets.resize(targetMin.size());

        this->alpha = learningRate;

        // All internal domains are [0,1]
        const int L = (int)Network.size() - 1;

        layers.reserve(L);

        for (int i = 0; i < L; ++i) {

            layers.emplace_back(
                Network[i + 1],
                Network[i],
                Grid[i],
                0.0, 1.0,
                0.0, 1.0,
                RND);
        }
    }

    // ---------------------------------------------------------
    // Normalize features and clip to [0,1]
    // ---------------------------------------------------------

    void NormalizeFeatures(const std::vector<double>& x) {

        if (x.size() != featureMin.size())
            throw std::invalid_argument("Feature size mismatch.");

        for (size_t i = 0; i < x.size(); ++i) {

            double v = (x[i] - featureMin[i]) /
                (featureMax[i] - featureMin[i]);

            if (v < 0.0) v = 0.0;
            if (v > 1.0) v = 1.0;
            normalizedFeatures[i] = v;
        }
    }

    // ---------------------------------------------------------
    // Normalize targets WITHOUT clipping
    // ---------------------------------------------------------

    void NormalizeTargets(const std::vector<double>& y) {

        if (y.size() != targetMin.size())
            throw std::invalid_argument("Target size mismatch.");

        for (size_t i = 0; i < y.size(); ++i) {

            normalizedTargets[i] =
                (y[i] - targetMin[i]) /
                (targetMax[i] - targetMin[i]);
        }
    }

    // ---------------------------------------------------------
    // Convert normalized predictions to original target ranges
    // ---------------------------------------------------------

    std::vector<double> DenormalizeTargets(
        const std::vector<double>& y) const {

        std::vector<double> result(y.size());

        for (size_t i = 0; i < y.size(); ++i) {

            result[i] = targetMin[i] +
                y[i] * (targetMax[i] - targetMin[i]);
        }

        return result;
    }

    // ---------------------------------------------------------
    // Forward (training)
    // ---------------------------------------------------------

    void Forward(const std::vector<double>& input) {

        NormalizeFeatures(input);

        layers[0].Forward(normalizedFeatures);

        for (size_t i = 1; i < layers.size(); ++i) {

            layers[i].Forward(
                layers[i - 1].response,
                i == layers.size() - 1);
        }
    }

    // ---------------------------------------------------------
    // Predict
    // ---------------------------------------------------------

    void Predict(const std::vector<double>& input) {

        NormalizeFeatures(input);

        layers[0].Predict(normalizedFeatures);

        for (size_t i = 1; i < layers.size(); ++i) {

            layers[i].Predict(
                layers[i - 1].response,
                i == layers.size() - 1);
        }
    }

    // ---------------------------------------------------------
    // Get prediction in original target units
    // ---------------------------------------------------------

    std::vector<double> GetPrediction() const {

        return DenormalizeTargets(layers.back().response);
    }

    // ---------------------------------------------------------
    // Update (training)
    // ---------------------------------------------------------

    void Update(const std::vector<double>& targets) {

        const int nLayers = (int)layers.size();

        NormalizeTargets(targets);

        // 1. Making derivatives

        for (int i = nLayers - 1; i > 0; --i) {

            layers[i].MakeDerivativeMatrix(
                layers[i - 1].adaptiveMin,
                layers[i - 1].adaptiveMax);
        }

        // 2. Top deltas

        const int nTargets = (int)normalizedTargets.size();

        for (int j = 0; j < nTargets; ++j) {

            layers[nLayers - 1].deltaS[j] =
                (normalizedTargets[j] -
                    layers[nLayers - 1].response[j]) * alpha;
        }

        // 3. Other deltas

        for (int i = nLayers - 1; i > 0; --i) {

            layers[i].BackPropStep(
                layers[i].deltaS,
                layers[i - 1].deltaS);
        }

        // 4. Updates

        for (int i = 0; i < nLayers; ++i) {
            layers[i].Update();
        }
    }

    // Single target overload

    void Update(const double target) {

        if (normalizedTargets.size() != 1)
            throw std::invalid_argument(
                "Scalar Update requires one output.");

        std::vector<double> targets(1);
        targets[0] = target;

        Update(targets);
    }

    // ---------------------------------------------------------
    // Normalize layers
    // ---------------------------------------------------------

    void NormalizeLayers() {

        const int nLayers = (int)layers.size();

        for (int i = 0; i < nLayers - 1; ++i) {
            layers[i].NormalizeLayer();
        }
    }

    // ---------------------------------------------------------
    // Accumulate
    // ---------------------------------------------------------

    void Accumulate(const KAN& kan) {

        const int nLayers = (int)this->layers.size();

        for (int i = 0; i < nLayers; ++i) {
            this->layers[i].Accumulate(kan.layers[i]);
        }
    }

    // ---------------------------------------------------------
    // Scale
    // ---------------------------------------------------------

    void Scale(double scale) {

        const int nLayers = (int)this->layers.size();

        for (int i = 0; i < nLayers; ++i) {
            this->layers[i].Scale(scale);
        }
    }

private:

    // =========================================================
    // Layer
    // =========================================================

    class Layer {
    public:

        Layer(int nUrysohns, int nFunctionsInU, int nPoints,
            double domainMin, double domainMax,
            double fmin, double fmax, std::mt19937& RNG) :

            response(nUrysohns),
            adaptiveMin(nUrysohns, fmin),
            adaptiveMax(nUrysohns, fmax),
            deltaS(nUrysohns),

            matrix(nUrysohns* nFunctionsInU,
                std::vector<double>(nPoints)),

            derivatives(nUrysohns,
                std::vector<double>(nFunctionsInU)),

            indexes(nUrysohns* nFunctionsInU),
            offsets(nUrysohns* nFunctionsInU),
            deltaX(nFunctionsInU),

            nUrysohns(nUrysohns),
            nFunctionsInU(nFunctionsInU),
            nPoints(nPoints),

            argMin(0.0),
            argDelta(0.0),
            targetMin(fmin),
            targetMax(fmax) {

            this->MakeRandomLayer(fmin, fmax, RNG);

            this->AssignMinAndDelta(
                domainMin, domainMax, nPoints);
        }

        inline void Forward(
            const std::vector<double>& input,
            bool returnAsIs = false) {

            for (int k = 0; k < nUrysohns; ++k) {
                response[k] =
                    this->OneUrysohn(input, k, returnAsIs);
            }
        }

        inline void Predict(
            const std::vector<double>& input,
            bool returnAsIs = false) {

            for (int k = 0; k < nUrysohns; ++k) {
                response[k] =
                    this->OneUrysohnPredict(input, k, returnAsIs);
            }
        }

        inline void MakeDerivativeMatrix(
            const std::vector<double>& min,
            const std::vector<double>& max) {

            for (int k = 0; k < nFunctionsInU; ++k) {
                deltaX[k] =
                    (max[k] - min[k]) / (nPoints - 1);
            }

            for (int k = 0; k < nUrysohns; ++k) {

                for (int j = 0; j < nFunctionsInU; ++j) {

                    derivatives[k][j] =
                        this->ComputeDerivative(
                            k * nFunctionsInU + j,
                            deltaX[j]);
                }
            }
        }

        inline void BackPropStep(
            const std::vector<double>& deltasIn,
            std::vector<double>& deltasOut) {

            for (int j = 0; j < nFunctionsInU; ++j) {

                deltasOut[j] = 0.0;

                for (int i = 0; i < nUrysohns; ++i) {

                    deltasOut[j] +=
                        derivatives[i][j] * deltasIn[i];
                }

                deltasOut[j] /= nUrysohns;
            }
        }

        inline void Update() {

            for (int k = 0; k < nUrysohns; ++k) {

                for (int j = 0; j < nFunctionsInU; ++j) {

                    this->UpdateOne(
                        deltaS[k],
                        k * nFunctionsInU + j);
                }
            }
        }

        void NormalizeLayer() {

            for (int b = 0; b < nUrysohns; ++b) {

                double ymin = adaptiveMin[b];
                double ymax = adaptiveMax[b];

                double s = (targetMax - targetMin) /
                    (ymax - ymin);

                double bias = targetMin - s * ymin;

                for (int j = 0; j < nFunctionsInU; ++j) {

                    auto& f = matrix[b * nFunctionsInU + j];

                    for (int k = 0; k < nPoints; ++k) {
                        f[k] = s * f[k] + bias;
                    }
                }

                adaptiveMin[b] = targetMin;
                adaptiveMax[b] = targetMax;
            }
        }

        void Accumulate(const Layer& layer) {

            for (size_t i = 0; i < matrix.size(); ++i) {

                for (size_t j = 0; j < matrix[i].size(); ++j) {

                    this->matrix[i][j] += layer.matrix[i][j];
                }
            }

            for (size_t i = 0; i < adaptiveMin.size(); ++i) {

                this->adaptiveMin[i] += layer.adaptiveMin[i];
                this->adaptiveMax[i] += layer.adaptiveMax[i];
            }
        }

        void Scale(double scale) {

            for (size_t i = 0; i < matrix.size(); ++i) {

                for (size_t j = 0; j < matrix[i].size(); ++j) {

                    this->matrix[i][j] *= scale;
                }
            }

            for (size_t i = 0; i < adaptiveMin.size(); ++i) {

                this->adaptiveMin[i] *= scale;
                this->adaptiveMax[i] *= scale;
            }
        }

        std::vector<double> response;
        std::vector<double> adaptiveMin;
        std::vector<double> adaptiveMax;
        std::vector<double> deltaS;

    private:

        inline double OneFunction(const double x, int id) {

            double R = (x - argMin) / argDelta;

            indexes[id] = (int)R;
            offsets[id] = R - indexes[id];

            assert(indexes[id] >= 0 &&
                indexes[id] <= (int)matrix[id].size() - 2);

            return matrix[id][indexes[id]] +
                (matrix[id][indexes[id] + 1] -
                    matrix[id][indexes[id]]) * offsets[id];
        }

        inline double OneUrysohn(
            const std::vector<double>& input,
            int nWhich, bool returnAsIs = false) {

            if ((int)input.size() != nFunctionsInU) {
                printf("Fatal: input size not equal Urysohn size");
                exit(0);
            }

            double r = 0.0;

            for (int j = 0; j < nFunctionsInU; ++j) {

                r += this->OneFunction(
                    input[j], nWhich * nFunctionsInU + j);
            }

            r /= nFunctionsInU;

            if (returnAsIs) {
                return r;
            }

            if (r > adaptiveMax[nWhich]) {
                adaptiveMax[nWhich] = r;
            }
            else if (r < adaptiveMin[nWhich]) {
                adaptiveMin[nWhich] = r;
            }

            return (targetMax - targetMin) *
                (r - adaptiveMin[nWhich]) /
                (adaptiveMax[nWhich] - adaptiveMin[nWhich])
                + targetMin;
        }

        inline double OneUrysohnPredict(
            const std::vector<double>& input,
            int nWhich, bool returnAsIs = false) {

            if ((int)input.size() != nFunctionsInU) {
                printf("Fatal: input size not equal Urysohn size");
                exit(0);
            }

            double r = 0.0;

            for (int j = 0; j < nFunctionsInU; ++j) {

                r += this->OneFunction(
                    input[j], nWhich * nFunctionsInU + j);
            }

            r /= nFunctionsInU;

            if (returnAsIs) {
                return r;
            }

            r = (targetMax - targetMin) *
                (r - adaptiveMin[nWhich]) /
                (adaptiveMax[nWhich] - adaptiveMin[nWhich])
                + targetMin;

            if (r < targetMin) return targetMin;
            if (r > targetMax) return targetMax;

            return r;
        }

        inline double ComputeDerivative(int id, double delta) {

            return (matrix[id][indexes[id] + 1] -
                matrix[id][indexes[id]]) / delta;
        }

        inline void MakeRandomLayer(
            double fmin, double fmax, std::mt19937& RND) {

            double mean = (fmax + fmin) / 2.0;
            double range = (fmax - fmin) / 4.0;

            double high = mean + range;
            double low = mean - range;

            std::uniform_real_distribution<double> dist(low, high);

            for (size_t i = 0; i < matrix.size(); ++i) {

                for (size_t j = 0; j < matrix[i].size(); ++j) {

                    matrix[i][j] = dist(RND);
                }
            }
        }

        inline void AssignMinAndDelta(
            double domainMin, double domainMax, int nPoints) {

            double gap = (domainMax - domainMin) * 0.05;

            argMin = domainMin - gap;

            argDelta =
                ((domainMax + gap) - argMin) / (nPoints - 1);
        }

        inline void UpdateOne(double delta, int id) {

            double tmp = delta * offsets[id];

            matrix[id][indexes[id] + 1] += tmp;
            matrix[id][indexes[id]] += delta - tmp;
        }

        std::vector<std::vector<double>> matrix;
        std::vector<std::vector<double>> derivatives;

        std::vector<int> indexes;
        std::vector<double> offsets;
        std::vector<double> deltaX;

        int nUrysohns, nFunctionsInU, nPoints;

        double argMin, argDelta, targetMin, targetMax;
    };

    // =========================================================
    // KAN members
    // =========================================================

    double alpha;

    // Original feature and target ranges
    std::vector<double> featureMin;
    std::vector<double> featureMax;

    std::vector<double> targetMin;
    std::vector<double> targetMax;

    // Reusable normalization buffers
    std::vector<double> normalizedFeatures;
    std::vector<double> normalizedTargets;

    std::vector<Layer> layers;
};
