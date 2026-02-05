/*
  Feature extraction functions for CBIR
*/
#ifndef FEATURES_H
#define FEATURES_H

#include <opencv2/opencv.hpp>
#include <vector>

// Task 1: Extract 7x7 center square as feature vector
int baseline_features(cv::Mat &src, std::vector<float> &features);

// Sum of Squared Differences distance metric
float ssd_distance(const std::vector<float> &f1, const std::vector<float> &f2);

#endif