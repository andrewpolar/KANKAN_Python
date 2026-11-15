
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <random>
#include <memory>
#include <utility>

#include "KAN.h"
#include "Ensemble.h"

namespace py = pybind11;


// Owns the data because EnsembleBuilder stores references to it.
class PyEnsembleBuilder {
private:
    std::vector<std::vector<double>> features;
    std::vector<std::vector<double>> targets;
    std::unique_ptr<EnsembleBuilder> builder;

public:
    PyEnsembleBuilder(
        std::vector<std::vector<double>> f,
        std::vector<std::vector<double>> t,
        int epochs)
        : features(std::move(f)),
        targets(std::move(t))
    {
        builder = std::make_unique<EnsembleBuilder>(
            features, targets, epochs
        );
    }

    KANEnsemble Build(
        const KAN& root,
        const std::vector<int>& indices,
        int depth,
        int mode,
        int initial_epochs)
    {
        return builder->Build(
            root, indices, depth, mode, initial_epochs
        );
    }
};


PYBIND11_MODULE(_kankan, m)
{
    m.doc() = "KANKAN: Kolmogorov-Arnold Network, Kaczmarz-Newton";

    py::class_<KAN>(m, "KAN")
        .def(py::init([](
            std::vector<int> network,
            std::vector<int> grid,
            const std::vector<double>& feature_min,
            const std::vector<double>& feature_max,
            const std::vector<double>& target_min,
            const std::vector<double>& target_max,
            double learning_rate,
            unsigned int seed)
            {
                std::mt19937 rng(seed);

                return std::make_unique<KAN>(
                    network, grid,
                    feature_min, feature_max,
                    target_min, target_max,
                    learning_rate, rng
                );
            }),
            py::arg("network"),
            py::arg("grid"),
            py::arg("feature_min"),
            py::arg("feature_max"),
            py::arg("target_min"),
            py::arg("target_max"),
            py::arg("learning_rate"),
            py::arg("seed") = 42)

        .def("forward", &KAN::Forward,
            py::call_guard<py::gil_scoped_release>())

        .def("update",
            py::overload_cast<const std::vector<double>&>(&KAN::Update),
            py::call_guard<py::gil_scoped_release>())

        .def("update",
            py::overload_cast<double>(&KAN::Update),
            py::call_guard<py::gil_scoped_release>())

        .def("predict", [](KAN& self, const std::vector<double>& input) {
        self.Predict(input);
        return self.GetPrediction();
            })

        .def("normalize_layers", &KAN::NormalizeLayers)
        .def("accumulate", &KAN::Accumulate)
        .def("scale", &KAN::Scale)

        .def("__copy__", [](const KAN& self) {
        return KAN(self);
            });


    py::class_<KANEnsemble>(m, "KANEnsemble")
        .def("predict_sample", &KANEnsemble::PredictSample);


    py::class_<PyEnsembleBuilder>(m, "EnsembleBuilder")
        .def(py::init<
            std::vector<std::vector<double>>,
            std::vector<std::vector<double>>,
            int
        >(),
            py::arg("features"),
            py::arg("targets"),
            py::arg("epochs_per_layer") = 1)

        .def("build", &PyEnsembleBuilder::Build,
            py::arg("root"),
            py::arg("indices"),
            py::arg("depth"),
            py::arg("mode"),
            py::arg("initial_epochs") = 8,
            py::call_guard<py::gil_scoped_release>());
}
