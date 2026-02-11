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

// Task 5: cosine distance for 512-D embeddings
float cosine_distance(const std::vector<float> &v1, const std::vector<float> &v2);

// Task 7
int custom_hsv_edge_features(const cv::Mat &src, std::vector<float> &features, int h_bins, int s_bins);
float edge_density(const cv::Mat &src);
float custom_distance(const std::vector<float> &target, const std::vector<float> &db, int h_bins, int s_bins);

// Extension
float banana_distance(const std::vector<float> &a,
                      const std::vector<float> &b);

float bluebin_distance(const std::vector<float> &a,
                       const std::vector<float> &b);

float face_metric_distance(const std::vector<float> &a,
                           const std::vector<float> &b);


#endif