/*
  Feature extraction functions for CBIR
*/
#ifndef FEATURES_H
#define FEATURES_H

#include <opencv2/opencv.hpp>
#include <vector>

// -------------------- Task 1 --------------------
int baseline_features(cv::Mat &src, std::vector<float> &features);
float ssd_distance(const std::vector<float> &f1, const std::vector<float> &f2);

// -------------------- Task 2 --------------------
int histogram_features(cv::Mat &src, std::vector<float> &features, int bins = 8);
float histogram_intersection(const std::vector<float> &h1, const std::vector<float> &h2);

// -------------------- Task 3 --------------------
int multi_histogram_features(cv::Mat &src, std::vector<float> &features, int bins = 8);
float multi_histogram_distance(const std::vector<float> &f1, const std::vector<float> &f2);

// -------------------- Task 4 --------------------
int texture_features(cv::Mat &src, std::vector<float> &features, int bins = 16);
int texture_color_features(cv::Mat &src, std::vector<float> &features,
                           int color_bins = 8, int texture_bins = 16);
float texture_color_distance(const std::vector<float> &f1, const std::vector<float> &f2,
                             int color_bins = 8, int texture_bins = 16);

// -------------------- Task 5 --------------------
float cosine_distance(const std::vector<float> &v1, const std::vector<float> &v2);

// -------------------- Task 7 --------------------
float edge_density(const cv::Mat &src);
int custom_hsv_edge_features(const cv::Mat &src, std::vector<float> &features,
                             int h_bins, int s_bins);
float custom_distance(const std::vector<float> &target, const std::vector<float> &db,
                      int h_bins, int s_bins);

// -------------------- Extensions --------------------

// Banana feature extractor (9-D banana blob stats used by banana.cpp + query.cpp)
// --- Banana extension ---
int banana_feature_vector(const cv::Mat &img, std::vector<float> &feat,
                          float hsv_s_min = 0.25f, float hsv_v_min = 0.25f);

// Banana distance (expects 9-D banana feature vectors)
float banana_distance(const std::vector<float> &a, const std::vector<float> &b);

// Blue bin distance (expects 9-D bluebin feature vectors)
float bluebin_distance(const std::vector<float> &a, const std::vector<float> &b);

// Face-aware metric distance (expects 1029-D face feature vectors)
float face_metric_distance(const std::vector<float> &a, const std::vector<float> &b);

int hsv_spatial_moments(const cv::Mat &src, std::vector<float> &feat, int grid=3);
float l2_distance(const std::vector<float> &a, const std::vector<float> &b);

int lbp_histogram(const cv::Mat &src, std::vector<float> &feat);
float chi_square_distance(const std::vector<float> &a, const std::vector<float> &b);

int moments_lbp_features(const cv::Mat &src, std::vector<float> &feat, int grid=3);
float moments_lbp_distance(const std::vector<float> &a, const std::vector<float> &b, int moments_len);

// --- Method B: LBP texture histogram (256 bins) ---
int lbp_histogram_features(const cv::Mat &src, std::vector<float> &features);
float chi_square_distance(const std::vector<float> &h1, const std::vector<float> &h2);


#endif
