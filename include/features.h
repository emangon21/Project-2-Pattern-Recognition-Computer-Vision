/*
  Feature extraction functions for CBIR
*/
#ifndef FEATURES_H
#define FEATURES_H

#include <opencv2/opencv.hpp>
#include <vector>

//Task 1: Extract 7x7 center square as feature vector
int baseline_features(cv::Mat &src, std::vector<float> &features);

//SSD distance metric
float ssd_distance(const std::vector<float> &f1, const std::vector<float> &f2);





//Task 2: Extract RGB color histogram
int histogram_features(cv::Mat &src, std::vector<float> &features, int bins = 8);

//Histogram intersection distance metric
float histogram_intersection(const std::vector<float> &h1, const std::vector<float> &h2);




//Task 3: Extract multi-histogram 
int multi_histogram_features(cv::Mat &src, std::vector<float> &features, int bins = 8);

// Multi-histogram distance with weighted combination
float multi_histogram_distance(const std::vector<float> &f1, const std::vector<float> &f2);



// Task 4: Extract texture - sobel magnitude histogram
int texture_features(cv::Mat &src, std::vector<float> &features, int bins = 16);

// Task 4: Extract combined texture and color features
int texture_color_features(cv::Mat &src, std::vector<float> &features, int color_bins = 8, int texture_bins = 16);

// Texture + Color distance with equal weighting
float texture_color_distance(const std::vector<float> &f1, const std::vector<float> &f2, int color_bins = 8, int texture_bins = 16);

#endif