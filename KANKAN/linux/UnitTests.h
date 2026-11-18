#pragma once
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cassert>
#include <chrono>
#include <thread>
#include "KAN.h"
#include "QuadraticDataHolder.h"
#include "Ensemble.h"

double Pearson(const std::vector<double>& x, const std::vector<double>& y) {
	int len = (int)x.size();
	double xmean = 0.0;
	double ymean = 0.0;
	for (int i = 0; i < len; ++i) {
		xmean += x[i];
		ymean += y[i];
	}
	xmean /= len;
	ymean /= len;

	double covariance = 0.0;
	for (int i = 0; i < len; ++i) {
		covariance += (x[i] - xmean) * (y[i] - ymean);
	}

	double stdX = 0.0;
	double stdY = 0.0;
	for (int i = 0; i < len; ++i) {
		stdX += (x[i] - xmean) * (x[i] - xmean);
		stdY += (y[i] - ymean) * (y[i] - ymean);
	}
	stdX = sqrt(stdX);
	stdY = sqrt(stdY);
	return covariance / stdX / stdY;
}

void ShowMatrix(const std::vector<std::vector<double>>& matrix) {
	size_t rows = matrix.size();
	size_t cols = matrix[0].size();
	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			printf("%5.3f ", matrix[i][j]);
		}
		printf("\n");
	}
}
void ShowVector(const std::vector<double>& ptr) {
	size_t N = ptr.size();
	int cnt = 0;
	for (size_t i = 0; i < N; ++i) {
		printf("%5.2f ", ptr[i]);
		if (++cnt >= 10) {
			printf("\n");
			cnt = 0;
		}
	}
}

///////////// Determinat dataset
std::vector<std::vector<double>> GenerateInput(int nRecords, int nFeatures, double min, double max, std::mt19937& RND) {
	std::uniform_real_distribution<double> dist(min, max);
	std::vector<std::vector<double>> x(nRecords);
	for (int i = 0; i < nRecords; ++i) {
		x[i] = std::vector<double>(nFeatures);
		for (int j = 0; j < nFeatures; ++j) {
			x[i][j] = dist(RND);
		}
	}
	return x;
}
double determinant(const std::vector<std::vector<double>>& matrix) {
	size_t n = (int)matrix.size();
	if (n == 1) {
		return matrix[0][0];
	}
	if (n == 2) {
		return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
	}
	double det = 0.0;
	for (size_t col = 0; col < n; ++col) {
		std::vector<std::vector<double>> subMatrix(n - 1, std::vector<double>(n - 1));
		for (size_t i = 1; i < n; ++i) {
			int subCol = 0;
			for (size_t j = 0; j < n; ++j) {
				if (j == col) continue;
				subMatrix[i - 1][subCol++] = matrix[i][j];
			}
		}
		det += (col % 2 == 0 ? 1 : -1) * matrix[0][col] * determinant(subMatrix);
	}
	return det;
}
double ComputeDeterminant(const std::vector<double>& input, int N) {
	std::vector<std::vector<double>> matrix(N, std::vector<double>(N, 0.0));
	int cnt = 0;
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			matrix[i][j] = input[cnt++];
		}
	}
	return determinant(matrix);
}
std::vector<double> ComputeDeterminantTarget(const std::vector<std::vector<double>>& x, int nMatrixSize) {
	size_t nRecords = (int)x.size();
	std::vector<double> target(nRecords);
	size_t counter = 0;
	while (true) {
		target[counter] = ComputeDeterminant(x[counter], nMatrixSize);
		if (++counter >= nRecords) break;
	}
	return target;
}
//End of determinant

///////// Areas of faces of tetrahedron
double Area(double x1, double y1, double z1, double x2, double y2, double z2, double x3, double y3, double z3) {
	double a1 = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1);
	double a2 = (x2 - x1) * (z3 - z1) - (z2 - z1) * (x3 - x1);
	double a3 = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
	double A = 0.5 * sqrt(a1 * a1 + a2 * a2 + a3 * a3);
	return A;
}
std::vector<std::vector<double>> MakeRandomMatrix(int rows, int cols, double min, double max, std::mt19937& RND) {
	std::uniform_real_distribution<double> dist(min, max);
	std::vector<std::vector<double>> matrix(rows);
	for (int i = 0; i < rows; ++i) {
		matrix[i] = std::vector<double>(cols);
		for (int j = 0; j < cols; ++j) {
			matrix[i][j] = dist(RND);
		}
	}
	return matrix;
}
std::vector<std::vector<double>> ComputeTargetMatrix(const std::vector<std::vector<double>>& X) {
	size_t rows = X.size();
	std::vector<std::vector<double>> matrix(rows);
	for (size_t i = 0; i < rows; ++i) {
		matrix[i] = std::vector<double>(4);
		matrix[i][0] = Area(X[i][0], X[i][1], X[i][2], X[i][3], X[i][4], X[i][5], X[i][6], X[i][7], X[i][8]);
		matrix[i][1] = Area(X[i][0], X[i][1], X[i][2], X[i][3], X[i][4], X[i][5], X[i][9], X[i][10], X[i][11]);
		matrix[i][2] = Area(X[i][0], X[i][1], X[i][2], X[i][6], X[i][7], X[i][8], X[i][9], X[i][10], X[i][11]);
		matrix[i][3] = Area(X[i][3], X[i][4], X[i][5], X[i][6], X[i][7], X[i][8], X[i][9], X[i][10], X[i][11]);
	}
	return matrix;
}
//////////// End tetrahedron

///////// Random triangles
std::vector<std::vector<double>> MakeRandomMatrixForTriangles(int rows, int cols, double min, double max, std::mt19937& RND) {
	std::uniform_real_distribution<double> dist(min, max);
	std::vector<std::vector<double>> matrix(rows);
	for (int i = 0; i < rows; ++i) {
		matrix[i] = std::vector<double>(cols);
		for (int j = 0; j < cols; ++j) {
			matrix[i][j] = dist(RND);
		}
	}
	return matrix;
}
double AreaOfTriangle(double x1, double y1, double x2, double y2, double x3, double y3) {
	double A = 0.5 * std::abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));
	return A;
}
std::vector<double> ComputeAreasOfTriangles(std::vector<std::vector<double>>& matrix) {
	size_t N = (int)matrix.size();
	std::vector<double> u(N);
	for (size_t i = 0; i < N; ++i) {
		u[i] = AreaOfTriangle(matrix[i][0], matrix[i][1], matrix[i][2], matrix[i][3], matrix[i][4], matrix[i][5]);
	}
	return u;
}
///////// End of random triangles

///////// Medians
double Median1(double x1, double y1, double x2, double y2, double x3, double y3) {
	double t1 = x1 - (x2 + x3) / 2.0;
	double t2 = y1 - (y2 + y3) / 2.0;
	t1 *= t1;
	t2 *= t2;
	return sqrt(t1 + t2);
}
double Median2(double x1, double y1, double x2, double y2, double x3, double y3) {
	double t1 = x2 - (x1 + x3) / 2.0;
	double t2 = y2 - (y1 + y3) / 2.0;
	t1 *= t1;
	t2 *= t2;
	return sqrt(t1 + t2);
}
double Median3(double x1, double y1, double x2, double y2, double x3, double y3) {
	double t1 = x3 - (x2 + x1) / 2.0;
	double t2 = y3 - (y2 + y1) / 2.0;
	t1 *= t1;
	t2 *= t2;
	return sqrt(t1 + t2);
}
std::vector<std::vector<double>> GenerateInputsMedians(int nRecords, int nFeatures, double min, double max, std::mt19937& RND) {
	std::uniform_real_distribution<double> dist(min, max);
	std::vector<std::vector<double>> x(nRecords);
	for (int i = 0; i < nRecords; ++i) {
		x[i] = std::vector<double>(nFeatures);
		for (int j = 0; j < nFeatures; ++j) {
			x[i][j] = dist(RND);
		}
	}
	return x;
}
std::vector<std::vector<double>> GenerateInputsMedians2(int nRecords, int nFeatures,
	double minX, double maxX, double minY, double maxY, std::mt19937& RND) {
	std::uniform_real_distribution<double> distx(minX, maxX);
	std::uniform_real_distribution<double> disty(minY, maxY);
	std::vector<std::vector<double>> x(nRecords, std::vector<double>(nFeatures));
	for (int i = 0; i < nRecords; ++i) {
		for (int j = 0; j < nFeatures; ++j) {
			if (j % 2 == 0) x[i][j] = distx(RND);
			else x[i][j] = disty(RND);
		}
	}
	return x;
}
std::vector<std::vector<double>> ComputeTargetsMedians(const std::vector<std::vector<double>>& x) {
	size_t nRecords = (int)x.size();
	std::vector<std::vector<double>> y(nRecords);
	for (size_t i = 0; i < nRecords; ++i) {
		y[i] = std::vector<double>(3);
		for (int j = 0; j < 3; ++j) {
			y[i][0] = Median1(x[i][0], x[i][1], x[i][2], x[i][3], x[i][4], x[i][5]);
			y[i][1] = Median2(x[i][0], x[i][1], x[i][2], x[i][3], x[i][4], x[i][5]);
			y[i][2] = Median3(x[i][0], x[i][1], x[i][2], x[i][3], x[i][4], x[i][5]);
		}
	}
	return y;
}
///////// End medians

std::pair<std::vector<double>, std::vector<double>>
getColumnMinMax(const std::vector<std::vector<double>>& matrix)
{
	if (matrix.empty() || matrix[0].empty())
		return {};

	const size_t cols = matrix[0].size();

	std::vector<double> mins = matrix[0];
	std::vector<double> maxs = matrix[0];

	for (const auto& row : matrix)
	{
		for (size_t j = 0; j < cols; ++j)
		{
			mins[j] = std::min(mins[j], row[j]);
			maxs[j] = std::max(maxs[j], row[j]);
		}
	}

	return { mins, maxs };
}

//Calibration for set of vectors. Computes median points and radius of hyper sphere covering 
//probability rate of points
std::pair<std::vector<double>, double> CalculateCalibration(const std::vector<std::vector<double>>& sample,
	double probability) {

	if (sample.empty())
		throw std::invalid_argument("Sample cannot be empty.");

	if (probability < 0.0 || probability > 1.0)
		throw std::out_of_range("Probability must be between 0 and 1.");

	const std::size_t dimensions = sample[0].size();

	// Check that all points have the same number of dimensions.
	for (const auto& point : sample) {
		if (point.size() != dimensions)
			throw std::invalid_argument("Inconsistent point dimensions.");
	}

	// Component-wise median.
	std::vector<double> centralPoint(dimensions);

	for (std::size_t d = 0; d < dimensions; ++d) {
		std::vector<double> values;
		values.reserve(sample.size());

		for (const auto& point : sample)
			values.push_back(point[d]);

		std::sort(values.begin(), values.end());

		const std::size_t middle = values.size() / 2;

		if (values.size() % 2 == 1)
			centralPoint[d] = values[middle];
		else
			centralPoint[d] =
			(values[middle - 1] + values[middle]) / 2.0;
	}

	// Distances from the central point.
	std::vector<double> distances;
	distances.reserve(sample.size());

	for (const auto& point : sample) {
		double sum = 0.0;

		for (std::size_t d = 0; d < dimensions; ++d)
		{
			const double diff = point[d] - centralPoint[d];
			sum += diff * diff;
		}

		distances.push_back(std::sqrt(sum));
	}

	std::sort(distances.begin(), distances.end());

	const std::size_t count = static_cast<std::size_t>(
		std::ceil(probability * sample.size()));

	const double radius = count == 0 ? 0.0 : distances[std::min(count - 1, distances.size() - 1)];
	return { centralPoint, radius };
}

//This demo shows how to split dataset into disjoints, train concurrently and 
//merge models. 
void DeterminantConcurrent() {
	const int nTrainingRecords = 100'000;
	const int nValidationRecords = 20'000;
	const int nMatrixSize = 4;
	const int nFeatures = nMatrixSize * nMatrixSize;
	const int nTargets = 1;
	const double alpha = 0.15;
	const int nDisjoints = 3;
	const int nDisjointSize = 38'000;
	const int nLoops = 12;
	const double minForFeatures = 0.0;
	const double maxForFeatures = 10.0;

	//Instantiate models
	std::random_device rd;
	std::mt19937 rng(rd());

	//making datasets
	auto features_training = GenerateInput(nTrainingRecords, nFeatures, minForFeatures, maxForFeatures, rng);
	auto features_validation = GenerateInput(nValidationRecords, nFeatures, minForFeatures, maxForFeatures, rng);
	auto targets_training = ComputeDeterminantTarget(features_training, nMatrixSize);
	auto targets_validation = ComputeDeterminantTarget(features_validation, nMatrixSize);

	auto resultF = getColumnMinMax(features_training);
	auto resultT = std::minmax_element(targets_training.begin(), targets_training.end());

	printf("Targets are determinants of 4 * 4 random matrices, %d training records\n", nTrainingRecords);
	printf("training is concurrent\n");
	try {
		auto tstart = std::chrono::high_resolution_clock::now();

		std::vector<KAN> kans;
		kans.reserve(nDisjoints);

		std::vector<double> targetMin(nTargets, *resultT.first);
		std::vector<double> targetMax(nTargets, *resultT.second);

		kans.emplace_back(KAN({ nFeatures, 70, nTargets }, { 4, 16 }, resultF.first, resultF.second,
			targetMin, targetMax, alpha, rng));

		for (int i = 1; i < nDisjoints; ++i) {
			kans.emplace_back(kans[0]);
		}

		std::vector<std::thread> threads;
		threads.reserve(nDisjoints);
		for (int loop = 0; loop < nLoops; ++loop) {
			threads.clear();
			for (int m = 0; m < nDisjoints; ++m) {
				threads.emplace_back([&, m]()
					{
						int start = (loop * nDisjoints + m) * nDisjointSize % nTrainingRecords;
						for (int i = 0; i < nDisjointSize; ++i) {
							int idx = (start + i) % nTrainingRecords;
							kans[m].Forward(features_training[idx]);
							kans[m].Update(targets_training[idx]);
						}
					});
			}
			for (auto& t : threads)
				t.join();

			for (int m = 1; m < nDisjoints; ++m)
				kans[0].Accumulate(kans[m]);

			kans[0].Scale(1.0 / (double)nDisjoints);

			for (int m = 1; m < nDisjoints; ++m)
				kans[m] = kans[0];

			auto tend = std::chrono::high_resolution_clock::now();
			auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart);

			printf("Loop %d of %d, time %lld MS\n", loop + 1, nLoops, (long long)ms.count());
		}

		auto tend = std::chrono::high_resolution_clock::now();
		auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart);
		printf("Training is completed for %lld MS\n", (long long)ms.count());

		//normalization of model
		kans[0].NormalizeLayers();

		std::vector<double> prediction(nValidationRecords);
		for (int record = 0; record < nValidationRecords; ++record) {
			kans[0].Predict(features_validation[record]);
			const auto& response = kans[0].GetPrediction();
			prediction[record] = response[0];
		}

		double pearson = Pearson(prediction, targets_validation);
		printf("Pearson for unseen data: %f\n\n", pearson);
	}
	catch (const std::exception& ex) {
		printf("%s\n", ex.what());
	}
	catch (...) {
		printf("Fatal\n");
	}
}

void TetrahedronConcurrent() {
	const int nTrainingRecords = 500'000;
	const int nValidationRecords = 50'000;
	const int nFeatures = 12;
	const int nTargets = 4;
	const double minForFeatures = 0.0;
	const double maxForFeatures = 10.0;
	//const double minForTargets = 10.0;
	//const double maxForTargets = 40.0;
	const double alpha = 0.2;
	const int nDisjoints = 10;
	const int nDisjointSize = 5'000;
	const int nLoops = 30;

	//Instantiate models
	std::random_device rd;
	std::mt19937 rng(rd());

	//generation
	auto features_training = MakeRandomMatrix(nTrainingRecords, nFeatures, minForFeatures, maxForFeatures, rng);
	auto features_validation = MakeRandomMatrix(nValidationRecords, nFeatures, minForFeatures, maxForFeatures, rng);
	auto targets_training = ComputeTargetMatrix(features_training);
	auto targets_validation = ComputeTargetMatrix(features_validation);

	auto resultF = getColumnMinMax(features_training);
	auto resultT = getColumnMinMax(targets_training);

	printf("Targets are areas of faces of random tetrahedrons, %d training records\n", nTrainingRecords);
	printf("training is concurrent\n");
	try {
		auto tstart = std::chrono::high_resolution_clock::now();

		std::vector<KAN> kans;
		kans.reserve(nDisjoints);

		kans.emplace_back(KAN({ nFeatures, 60, 10 , nTargets }, { 2, 12, 22 }, resultF.first, resultF.second,
			resultT.first, resultT.second, alpha, rng));

		for (int i = 1; i < nDisjoints; ++i) {
			kans.emplace_back(kans[0]);
		}

		std::vector<std::thread> threads;
		threads.reserve(nDisjoints);
		for (int loop = 0; loop < nLoops; ++loop) {
			threads.clear();
			for (int m = 0; m < nDisjoints; ++m) {
				threads.emplace_back([&, m]()
					{
						int start = (loop * nDisjoints + m) * nDisjointSize % nTrainingRecords;
						for (int i = 0; i < nDisjointSize; ++i) {
							int idx = (start + i) % nTrainingRecords;
							kans[m].Forward(features_training[idx]);
							kans[m].Update(targets_training[idx]);
						}
					});
			}
			for (auto& t : threads)
				t.join();

			for (int m = 1; m < nDisjoints; ++m)
				kans[0].Accumulate(kans[m]);

			kans[0].Scale(1.0 / (double)nDisjoints);

			for (int m = 1; m < nDisjoints; ++m)
				kans[m] = kans[0];

			auto tend = std::chrono::high_resolution_clock::now();
			auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart);

			printf("Loop %d of %d, time %lld MS\n", loop + 1, nLoops, (long long)ms.count());
		}

		auto tend = std::chrono::high_resolution_clock::now();
		auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart);
		printf("Training is completed for %lld MS\n", (long long)ms.count());

		//normalization of model
		kans[0].NormalizeLayers();

		//validation
		std::vector<double> prediction1(nValidationRecords);
		std::vector<double> prediction2(nValidationRecords);
		std::vector<double> prediction3(nValidationRecords);
		std::vector<double> prediction4(nValidationRecords);
		//
		std::vector<double> target1(nValidationRecords);
		std::vector<double> target2(nValidationRecords);
		std::vector<double> target3(nValidationRecords);
		std::vector<double> target4(nValidationRecords);
		//
		for (int record = 0; record < nValidationRecords; ++record) {
			kans[0].Predict(features_validation[record]);
			const auto& response = kans[0].GetPrediction();
			prediction1[record] = response[0];
			prediction2[record] = response[1];
			prediction3[record] = response[2];
			prediction4[record] = response[3];
			//
			target1[record] = targets_validation[record][0];
			target2[record] = targets_validation[record][1];
			target3[record] = targets_validation[record][2];
			target4[record] = targets_validation[record][3];
		}

		double p1 = Pearson(prediction1, target1);
		double p2 = Pearson(prediction2, target2);
		double p3 = Pearson(prediction3, target3);
		double p4 = Pearson(prediction4, target4);
		printf("Pearsons for unseen data: %f, %f, %f, %f\n\n",
			p1, p2, p3, p4);
	}
	catch (const std::exception& ex) {
		printf("%s\n", ex.what());
	}
	catch (...) {
		printf("Fatal\n");
	}
}

void TrianglesSequential() {
	//parameters of dataset
	const int nTargets = 1;
	const int nFeatures = 6;
	const int nTrainingRecords = 10'000;
	const int nValidationRecords = 2'000;
	const double minForFeatures = 0.0;
	const double maxForFeatures = 10.0;
	const double alpha = 0.05;
	const int nEpochs = 16;

	std::random_device rd;
	std::mt19937 rng(rd());

	//making datasets
	auto features_training = MakeRandomMatrixForTriangles(nTrainingRecords, nFeatures, minForFeatures, maxForFeatures, rng);
	auto features_validation = MakeRandomMatrixForTriangles(nValidationRecords, nFeatures, minForFeatures, maxForFeatures, rng);
	auto targets_training = ComputeAreasOfTriangles(features_training);
	auto targets_validation = ComputeAreasOfTriangles(features_validation);

	auto resultF = getColumnMinMax(features_training);
	auto resultT = std::minmax_element(targets_training.begin(), targets_training.end());

	printf("Targets are areas of random triangles, %d training records\n", nTrainingRecords);
	printf("training is sequential ...\n");
	try {
		auto tstart = std::chrono::high_resolution_clock::now();

		std::vector<double> targetMin(nTargets, *resultT.first);
		std::vector<double> targetMax(nTargets, *resultT.second);

		KAN kan({ nFeatures, 50, 10, 4, nTargets }, { 2, 12, 12, 22 }, resultF.first, resultF.second,
			targetMin, targetMax, alpha, rng);

		for (int epoch = 0; epoch < nEpochs; ++epoch) {
			for (int record = 0; record < nTrainingRecords; ++record) {
				kan.Forward(features_training[record]);
				kan.Update(targets_training[record]);
			}
			printf("%d ", epoch + 1);
		}
		printf("\n");

		auto tend = std::chrono::high_resolution_clock::now();
		auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart);
		printf("Training is completed for %lld MS\n", (long long)ms.count());

		kan.NormalizeLayers();

		auto prediction = std::vector<double>(nValidationRecords);
		for (int record = 0; record < nValidationRecords; ++record) {
			kan.Predict(features_validation[record]);
			const auto& response = kan.GetPrediction();
			prediction[record] = response[0];
		}

		double p = Pearson(prediction, targets_validation);
		printf("Pearson for unseen data: %2.4f\n\n", p);
	}
	catch (const std::exception& ex) {
		printf("%s\n", ex.what());
	}
	catch (...) {
		printf("Fatal\n");
	}
}

void MediansSequential() {
	//data
	const int nTrainingRecords = 10'000;
	const int nValidationRecords = 2'000;
	const int nFeatures = 6;
	const int nTargets = 3;
	const double minForFeatures = 0.0;
	const double maxForFeatures = 10.0;
	const double alpha = 0.2;
	const int nEpochs = 16;

	std::random_device rd;
	std::mt19937 rng(rd());

	//data generation
	auto features_training = GenerateInputsMedians(nTrainingRecords, nFeatures, minForFeatures, maxForFeatures, rng);
	auto features_validation = GenerateInputsMedians(nValidationRecords, nFeatures, minForFeatures, maxForFeatures, rng);
	auto targets_training = ComputeTargetsMedians(features_training);
	auto targets_validation = ComputeTargetsMedians(features_validation);

	auto resultF = getColumnMinMax(features_training);
	auto resultT = getColumnMinMax(targets_training);

	printf("Targets are medians of random triangles, %d training records\n", nTrainingRecords);
	printf("training is sequential ...\n");
	try {
		auto tstart = std::chrono::high_resolution_clock::now();

		KAN kan({ nFeatures, 20, 10, 4, nTargets }, { 2, 12, 12, 22 }, resultF.first, resultF.second,
			resultT.first, resultT.second, alpha, rng);

		for (int epoch = 0; epoch < nEpochs; ++epoch) {
			for (int record = 0; record < nTrainingRecords; ++record) {
				kan.Forward(features_training[record]);
				kan.Update(targets_training[record]);
			}
			printf("%d ", epoch + 1);
		}
		printf("\n");

		auto tend = std::chrono::high_resolution_clock::now();
		auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart);
		printf("Training is completed for %lld MS\n", (long long)ms.count());

		kan.NormalizeLayers();

		auto prediction0 = std::vector<double>(nValidationRecords);
		auto prediction1 = std::vector<double>(nValidationRecords);
		auto prediction2 = std::vector<double>(nValidationRecords);
		//
		auto target0 = std::vector<double>(nValidationRecords);
		auto target1 = std::vector<double>(nValidationRecords);
		auto target2 = std::vector<double>(nValidationRecords);
		//
		for (int record = 0; record < nValidationRecords; ++record) {
			kan.Predict(features_validation[record]);
			const auto& response = kan.GetPrediction();
			//
			prediction0[record] = response[0];
			prediction1[record] = response[1];
			prediction2[record] = response[2];
			//
			target0[record] = targets_validation[record][0];
			target1[record] = targets_validation[record][1];
			target2[record] = targets_validation[record][2];
		}

		double p0 = Pearson(prediction0, target0);
		double p1 = Pearson(prediction1, target1);
		double p2 = Pearson(prediction2, target2);
		printf("Pearsons for unseen data: %2.4f %2.4f %2.4f\n\n", p0, p1, p2);

		//kan.ShowVectorMappings();
		//kan.ShowFunctionsComplexity();
		//kan.ShowLimitsInLayers();
		//kan.ShowMatrices();
	}
	catch (const std::exception& ex) {
		printf("%s\n", ex.what());
	}
	catch (...) {
		printf("Fatal\n");
	}
}

void Medians2Sequential() {
	//data
	const int nTrainingRecords = 10'000;
	const int nValidationRecords = 2'000;
	const int nFeatures = 6;
	const int nTargets = 3;
	const double minForFeaturesX = 1.0;
	const double maxForFeaturesX = 10.0;
	const double minForFeaturesY = -10.0;
	const double maxForFeaturesY = -1.0;
	const double alpha = 0.2;
	const int nEpochs = 16;

	std::random_device rd;
	std::mt19937 rng(rd());

	//data generation with different ranges 
	auto features_training = GenerateInputsMedians2(nTrainingRecords, nFeatures, 
		minForFeaturesX, maxForFeaturesX, minForFeaturesY, maxForFeaturesY, rng);
	//
	auto features_validation = GenerateInputsMedians2(nValidationRecords, nFeatures, 
		minForFeaturesX, maxForFeaturesX, minForFeaturesY, maxForFeaturesY, rng);
	//
	auto targets_training = ComputeTargetsMedians(features_training);
	auto targets_validation = ComputeTargetsMedians(features_validation);

	auto resultF = getColumnMinMax(features_training);
	auto resultT = getColumnMinMax(targets_training);

	printf("Targets are medians of random triangles, %d training records\n", nTrainingRecords);
	printf("features have different ranges, training is sequential ...\n");
	try {
		auto tstart = std::chrono::high_resolution_clock::now();

		KAN kan({ nFeatures, 20, 10, 4, nTargets }, { 2, 12, 12, 22 }, resultF.first, resultF.second,
			resultT.first, resultT.second, alpha, rng);

		for (int epoch = 0; epoch < nEpochs; ++epoch) {
			for (int record = 0; record < nTrainingRecords; ++record) {
				kan.Forward(features_training[record]);
				kan.Update(targets_training[record]);
			}
			printf("%d ", epoch + 1);
		}
		printf("\n");

		auto tend = std::chrono::high_resolution_clock::now();
		auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart);
		printf("Training is completed for %lld MS\n", (long long)ms.count());

		kan.NormalizeLayers();

		auto prediction0 = std::vector<double>(nValidationRecords);
		auto prediction1 = std::vector<double>(nValidationRecords);
		auto prediction2 = std::vector<double>(nValidationRecords);
		//
		auto target0 = std::vector<double>(nValidationRecords);
		auto target1 = std::vector<double>(nValidationRecords);
		auto target2 = std::vector<double>(nValidationRecords);
		//
		for (int record = 0; record < nValidationRecords; ++record) {
			kan.Predict(features_validation[record]);
			const auto& response = kan.GetPrediction();
			//
			prediction0[record] = response[0];
			prediction1[record] = response[1];
			prediction2[record] = response[2];
			//
			target0[record] = targets_validation[record][0];
			target1[record] = targets_validation[record][1];
			target2[record] = targets_validation[record][2];
		}

		double p0 = Pearson(prediction0, target0);
		double p1 = Pearson(prediction1, target1);
		double p2 = Pearson(prediction2, target2);
		printf("Pearsons for unseen data: %2.4f %2.4f %2.4f\n\n", p0, p1, p2);
	}
	catch (const std::exception& ex) {
		printf("%s\n", ex.what());
	}
	catch (...) {
		printf("Fatal\n");
	}
}

//Unit test of stochastic model
void TestMDDR() {
	constexpr int n = 10;
	constexpr int N = 10000;
	constexpr int TreeDepth = 8;

	constexpr double minForTargets = -100.0;
	constexpr double maxForTargets = 100.0;
	constexpr double alpha = 0.2;

	constexpr int nEpochsPerLayer = 1;
	constexpr int nInitialEpochs = 8;
	constexpr int nTargets = 4;
	constexpr int nFeatures = 8;

	// Generate a fresh random seed for this run.
	std::random_device rd;
	std::mt19937 rng(rd());

	// Generate training data.
	QuadraticDataHolder DH(n, nFeatures, nTargets, minForTargets, maxForTargets, rng());
	DH.GenerateData(N);

	auto resultF = getColumnMinMax(DH.GetFeatures());
	auto resultT = getColumnMinMax(DH.GetTargets());

	try {
		// Network architecture and interpolation points.
		std::vector<int> network = { nFeatures, 16, nTargets };
		std::vector<int> points = { 4, 4 };

		KAN rootKan(network, points, resultF.first, resultF.second, resultT.first, resultT.second,
			alpha, rng);

		// Initial training-record indices: 0 .. N-1.
		std::vector<int> indices(N);
		std::iota(indices.begin(), indices.end(), 0);

		// Build and train the stochastic KAN ensemble.
		EnsembleBuilder builder(DH.GetFeatures(), DH.GetTargets(), nEpochsPerLayer);
		//EnsembleBuilder builder(DH.GetFeatures(), DH.GetTargets());  //nEpochsPerLayer defaulted to 1

		const auto start = std::chrono::steady_clock::now();

		printf("Testing of stochastic system\n");
		printf("Features are 8 quantities of dice, 8 hidden random variables are sums of roll outcomes,\n");
		printf("4 targets are quadratic forms.\n");
		printf("The model is ensemble of KANs.\n");
		printf("Training:\n");

		KANEnsemble ensemble = builder.Build(rootKan, indices, TreeDepth, 2, nInitialEpochs);
		//KANEnsemble ensemble = builder.Build(rootKan, indices, TreeDepth, 2); //nInitialEpochs defaulted to 8
		const auto stop = std::chrono::steady_clock::now();
		const std::chrono::duration<double> elapsed = stop - start;

		printf("Training time: %3.4f seconds\n", elapsed.count());

		using Clock = std::chrono::steady_clock;
		auto start2 = Clock::now();
		printf("\nCalibration of MDDR model:\n");

		for (int step = 1; step <= 9; ++step) {
			const double P = 0.10 * step;

			int within = 0;
			int total = 0;

			for (int record = 0; record < N; ++record) {
				// Generate a new, unseen record.
				std::vector<double> feature = DH.GenerateFeature();

				std::vector<double> hidden = DH.GenerateHidden(feature);
				std::vector<double> target = DH.GenerateTarget(hidden);
				DH.NormalizeTarget(target);

				// Predict using the ensemble.
				std::vector<std::vector<double>> sample = ensemble.PredictSample(feature);

				// Calculate prediction point and radius.
				auto calibration = CalculateCalibration(sample, P);
				const auto& point = calibration.first;
				double radius = calibration.second;

				// Calculate Euclidean distance from actual target.
				double distanceSquared = 0.0;

				for (std::size_t j = 0; j < point.size(); ++j) {
					const double difference = point[j] - target[j];
					distanceSquared += difference * difference;
				}

				const double distance = std::sqrt(distanceSquared);

				if (distance <= radius) ++within;
				++total;
			}

			const double actual = static_cast<double>(within) / total;
			printf("Target %3.2f, actual %3.2f\n", P, actual);
		}

		auto stop2 = Clock::now();
		const std::chrono::duration<double> elapsed2 = stop2 - start2;
		printf("MDDR calibration time: %3.4f seconds\n\n", elapsed2.count());
	}
	catch (const std::exception& ex) {
		printf("%s\n", ex.what());
	}
	catch (...) {
		printf("Fatal\n");
	}
}
