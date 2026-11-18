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

        this->featureMin = featureMin;
        this->featureMax = featureMax;
        this->targetMin = targetMin;
        this->targetMax = targetMax;

        for (size_t i = 0; i < featureMin.size(); ++i) {
            if (featureMax[i] <= featureMin[i])
                throw std::invalid_argument("Invalid feature range.");
        }

        for (size_t i = 0; i < targetMin.size(); ++i) {
            if (targetMax[i] <= targetMin[i])
                throw std::invalid_argument("Invalid target range.");
        }

        normalizedFeatures.resize(featureMin.size());
        normalizedTargets.resize(targetMin.size());
        this->alpha = learningRate;
        const int L = (int)Network.size() - 1;
        layers.reserve(L);
        for (int i = 0; i < L; ++i) {
            layers.emplace_back(Network[i + 1], Network[i], Grid[i], 0.0, 1.0, 0.0, 1.0, RND);
        }
    }

    void NormalizeFeatures(const std::vector<double>& x) {
        if (x.size() != featureMin.size())
            throw std::invalid_argument("Feature size mismatch.");

        for (size_t i = 0; i < x.size(); ++i) {
            double v = (x[i] - featureMin[i]) / (featureMax[i] - featureMin[i]);
            if (v < 0.0) v = 0.0;
            if (v > 1.0) v = 1.0;
            normalizedFeatures[i] = v;
        }
    }

    void NormalizeTargets(const std::vector<double>& y) {
        if (y.size() != targetMin.size())
            throw std::invalid_argument("Target size mismatch.");

        for (size_t i = 0; i < y.size(); ++i) {
            normalizedTargets[i] = (y[i] - targetMin[i]) / (targetMax[i] - targetMin[i]);
        }
    }

    std::vector<double> DenormalizeTargets(const std::vector<double>& y) const {
        std::vector<double> result(y.size());
        for (size_t i = 0; i < y.size(); ++i) {
            result[i] = targetMin[i] + y[i] * (targetMax[i] - targetMin[i]);
        }
        return result;
    }

    void Forward(const std::vector<double>& input) {
        NormalizeFeatures(input);
        layers[0].Forward(normalizedFeatures);
        for (size_t i = 1; i < layers.size(); ++i) {
            layers[i].Forward( layers[i - 1].response, i == layers.size() - 1);
        }
    }

    void Predict(const std::vector<double>& input) {
        NormalizeFeatures(input);
        layers[0].Predict(normalizedFeatures);
        for (size_t i = 1; i < layers.size(); ++i) {
            layers[i].Predict(layers[i - 1].response, i == layers.size() - 1);
        }
    }

    std::vector<double> GetPrediction() const {
        return DenormalizeTargets(layers.back().response);
    }

    void Update(const std::vector<double>& targets) {
        const int nLayers = (int)layers.size();
        NormalizeTargets(targets);

        // 1. Making derivatives
        for (int i = nLayers - 1; i > 0; --i) {
            layers[i].MakeDerivativeMatrix(layers[i - 1].deltaAbscissa);
        }

        // 2. Top deltas
        const int nTargets = (int)normalizedTargets.size();
        for (int j = 0; j < nTargets; ++j) {
            layers[nLayers - 1].deltaS[j] = (normalizedTargets[j] - layers[nLayers - 1].response[j]) * alpha;
        }

        // 3. Other deltas
        for (int i = nLayers - 1; i > 0; --i) {
            layers[i].BackPropStep(layers[i].deltaS, layers[i - 1].deltaS);
        }

        // 4. Updates
        for (int i = 0; i < nLayers; ++i) {
            layers[i].Update();
        }
    }

    void Update(const double target) {
        if (normalizedTargets.size() != 1)
            throw std::invalid_argument(
                "Scalar Update requires one output.");

        std::vector<double> targets(1);
        targets[0] = target;
        Update(targets);
    }

    void NormalizeLayers() {
        const int nLayers = (int)layers.size();
        for (int i = 0; i < nLayers - 1; ++i) {
            layers[i].NormalizeLayer();
        }
    }

    void Accumulate(const KAN& kan) {
        const int nLayers = (int)this->layers.size();
        for (int i = 0; i < nLayers; ++i) {
            this->layers[i].Accumulate(kan.layers[i]);
        }
    }

    void Scale(double scale) {
        const int nLayers = (int)this->layers.size();
        for (int i = 0; i < nLayers; ++i) {
            this->layers[i].Scale(scale);
        }
    }

    void ShowVectorMappings() {
        const int nLayers = (int)this->layers.size();
        printf("Number of features %d\n", (int)featureMin.size());
        for (int i = 0; i < nLayers; ++i) {
            printf("Layer %d, output size %d\n", i, this->layers[i].OutputSize());
        }
        printf("\n");
    }

    void ShowFunctionsComplexity() {
        const int nLayers = (int)this->layers.size();
        for (int i = 0; i < nLayers; ++i) {
            printf("Layer %d, points in functions %d\n", i, this->layers[i].nPointsInFunctions());
        }
        printf("\n");
    }

    void ShowLimitsInLayers() {
        const int nLayers = (int)this->layers.size();
        for (int i = 0; i < nLayers; ++i) {
            printf("Layer %d\n", i);
            this->layers[i].ShowActualLimits();
        }
        printf("\n");
    }

    void ShowMatrices() {
        const int nLayers = (int)this->layers.size();
        for (int i = 0; i < nLayers; ++i) {
            printf("Layer %d\n", i);
            this->layers[i].ShowAllU();
        }
        printf("\n");
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

            // Public members, in declaration order
            response(nUrysohns),
            deltaS(nUrysohns),
            deltaAbscissa(nUrysohns),

            // Private members, in declaration order
            matrix(nUrysohns* nFunctionsInU, std::vector<double>(nPoints)),
            derivatives(nUrysohns, std::vector<double>(nFunctionsInU)),
            indexes(nUrysohns* nFunctionsInU),
            offsets(nUrysohns* nFunctionsInU),
            deltaX(nFunctionsInU),

            nUrysohns(nUrysohns),
            nFunctionsInU(nFunctionsInU),
            nPoints(nPoints),

            argMin(0.0),
            argDelta(0.0),
            targetMin(fmin),
            targetMax(fmax),

            adaptiveMin(nUrysohns, fmin),
            adaptiveMax(nUrysohns, fmax)
        {
            this->MakeRandomLayer(fmin, fmax, RNG);
            this->AssignMinAndDelta(domainMin, domainMax, nPoints);
        }

        inline void Forward(const std::vector<double>& input, bool returnAsIs = false) {
            for (int k = 0; k < nUrysohns; ++k) {
                response[k] = this->OneUrysohn(input, k, returnAsIs);
            }
        }

        inline void Predict(const std::vector<double>& input, bool returnAsIs = false) {
            for (int k = 0; k < nUrysohns; ++k) {
                response[k] = this->OneUrysohnPredict(input, k, returnAsIs);
            }
        }

        inline void MakeDerivativeMatrix(const std::vector<double>& deltaX) {
            for (int k = 0; k < nUrysohns; ++k) {
                for (int j = 0; j < nFunctionsInU; ++j) {
                    derivatives[k][j] = this->ComputeDerivative( k * nFunctionsInU + j, deltaX[j]);
                }
            }
        }

        inline void BackPropStep(const std::vector<double>& deltasIn, std::vector<double>& deltasOut) {
            for (int j = 0; j < nFunctionsInU; ++j) {
                deltasOut[j] = 0.0;
                for (int i = 0; i < nUrysohns; ++i) {
                    deltasOut[j] += derivatives[i][j] * deltasIn[i];
                }
                deltasOut[j] /= nUrysohns;
            }
        }

        inline void Update() {
            for (int k = 0; k < nUrysohns; ++k) {
                for (int j = 0; j < nFunctionsInU; ++j) {
                    this->UpdateOne(deltaS[k], k * nFunctionsInU + j);
                }
            }
        }

        void NormalizeLayer() {
            for (int b = 0; b < nUrysohns; ++b) {
                double ymin = adaptiveMin[b];
                double ymax = adaptiveMax[b];
                double s = (targetMax - targetMin) / (ymax - ymin);
                double bias = targetMin - s * ymin;
                for (int j = 0; j < nFunctionsInU; ++j) {
                    auto& f = matrix[b * nFunctionsInU + j];
                    for (int k = 0; k < nPoints; ++k) {
                        f[k] = s * f[k] + bias;
                    }
                }
                adaptiveMin[b] = targetMin;
                adaptiveMax[b] = targetMax;
                deltaAbscissa[b] = (adaptiveMax[b] - adaptiveMin[b]) / (nPoints - 1);
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

        //Printout parameters
        int OutputSize() {
            return nUrysohns;
        }
        
        int nPointsInFunctions() {
            return nPoints;
        }

        void ShowActualLimits() {
            for (size_t i = 0; i < adaptiveMin.size(); ++i) {
                printf("%6.4f ", adaptiveMin[i]);
            }
            printf("\n");
            for (size_t i = 0; i < adaptiveMax.size(); ++i) {
                printf("%6.4f ", adaptiveMax[i]);
            }
            printf("\n");
        }

        void ShowU(int nWhich) {
            for (int i = 0; i < nFunctionsInU; ++i) {
                for (int j = 0; j < nPoints; ++j) {
                    printf("%6.4f ", matrix[nWhich * nFunctionsInU + i][j]);
                }
                printf("\n");
            }
            printf("\n");
        }

        void ShowAllU() {
            for (int i = 0; i < nUrysohns; ++i) {
                ShowU(i);
            }
        }

        std::vector<double> response;
        std::vector<double> deltaS;
        std::vector<double> deltaAbscissa;

    private:

        inline double OneFunction(const double x, int id) {
            double R = (x - argMin) / argDelta;
            indexes[id] = (int)R;
            offsets[id] = R - indexes[id];
            assert(indexes[id] >= 0 && indexes[id] <= (int)matrix[id].size() - 2);
            return matrix[id][indexes[id]] + (matrix[id][indexes[id] + 1] -
                    matrix[id][indexes[id]]) * offsets[id];
        }

        inline double OneUrysohn(const std::vector<double>& input, int nWhich, bool returnAsIs = false) {
            if ((int)input.size() != nFunctionsInU) {
                printf("Fatal: input size not equal Urysohn size");
                exit(0);
            }
            double r = 0.0;
            for (int j = 0; j < nFunctionsInU; ++j) {
                r += this->OneFunction(input[j], nWhich * nFunctionsInU + j);
            }
            r /= nFunctionsInU;
            if (returnAsIs) {
                return r;
            }
            if (r > adaptiveMax[nWhich]) {
                adaptiveMax[nWhich] = r;
                deltaAbscissa[nWhich] = (adaptiveMax[nWhich] - adaptiveMin[nWhich]) / (nPoints - 1);
            }
            else if (r < adaptiveMin[nWhich]) {
                adaptiveMin[nWhich] = r;
                deltaAbscissa[nWhich] = (adaptiveMax[nWhich] - adaptiveMin[nWhich]) / (nPoints - 1);
            }
            return (targetMax - targetMin) * (r - adaptiveMin[nWhich]) /
                (adaptiveMax[nWhich] - adaptiveMin[nWhich]) + targetMin;
        }

        inline double OneUrysohnPredict(const std::vector<double>& input, int nWhich, bool returnAsIs = false) {

            if ((int)input.size() != nFunctionsInU) {
                printf("Fatal: input size not equal Urysohn size");
                exit(0);
            }
            double r = 0.0;
            for (int j = 0; j < nFunctionsInU; ++j) {
                r += this->OneFunction(input[j], nWhich * nFunctionsInU + j);
            }
            r /= nFunctionsInU;
            if (returnAsIs) {
                return r;
            }
            r = (targetMax - targetMin) * (r - adaptiveMin[nWhich]) /
                (adaptiveMax[nWhich] - adaptiveMin[nWhich]) + targetMin;

            if (r < targetMin) return targetMin;
            if (r > targetMax) return targetMax;
            return r;
        }

        inline double ComputeDerivative(int id, double delta) {
            return (matrix[id][indexes[id] + 1] - matrix[id][indexes[id]]) / delta;
        }

        inline void MakeRandomLayer(double fmin, double fmax, std::mt19937& RND) {
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

        inline void AssignMinAndDelta(double domainMin, double domainMax, int nPoints) {
            double gap = (domainMax - domainMin) * 0.05;
            argMin = domainMin - gap;
            argDelta = ((domainMax + gap) - argMin) / (nPoints - 1);
            for (size_t i = 0; i < adaptiveMax.size(); ++i) {
                deltaAbscissa[i] = (adaptiveMax[i] - adaptiveMin[i]) / (nPoints - 1);
            }
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
        std::vector<double> adaptiveMin;
        std::vector<double> adaptiveMax;
    };

    // =========================================================
    // KAN members
    // =========================================================

    double alpha;
    std::vector<double> featureMin;
    std::vector<double> featureMax;
    std::vector<double> targetMin;
    std::vector<double> targetMax;
    std::vector<double> normalizedFeatures;
    std::vector<double> normalizedTargets;
    std::vector<Layer> layers;
};
