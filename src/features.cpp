/*
  Feature extraction implementations
*/
#include "features.h"
#include <cmath>
#include <iostream>

//Task 1: Extract 7x7 center square as baseline feature
int baseline_features(cv::Mat &src, std::vector<float> &features) {
    features.clear();
    
    //center of image
    int center_row = src.rows / 2;
    int center_col = src.cols / 2;
    
    //extract 7x7 region around center (3 pixels on each side)
    int half_size = 3;
    
    //is image large enough
    if (center_row - half_size < 0 || center_row + half_size >= src.rows ||
        center_col - half_size < 0 || center_col + half_size >= src.cols) {
        std::cerr << "Error: Image too small for 7x7 center extraction" << std::endl;
        return -1;
    }
    
    //extract 7x7 region row by row
    for (int i = -half_size; i <= half_size; i++) {
        for (int j = -half_size; j <= half_size; j++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(center_row + i, center_col + j);
            // Store B, G, R values
            features.push_back(static_cast<float>(pixel[0])); // B
            features.push_back(static_cast<float>(pixel[1])); // G
            features.push_back(static_cast<float>(pixel[2])); // R
        }
    }
    
    //7*7*3 = 147 features
    return 0;
}

//sum of squared differences metric
float ssd_distance(const std::vector<float> &f1, const std::vector<float> &f2) {
    if (f1.size() != f2.size()) {
        std::cerr << "Error: Feature vectors must be same size" << std::endl;
        return -1.0f;
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < f1.size(); i++) {
        float diff = f1[i] - f2[i];
        sum += diff * diff;
    }
    return sum;
}