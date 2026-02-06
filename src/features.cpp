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


//Task 2: Extract RGB color histogram
int histogram_features(cv::Mat &src, std::vector<float> &features, int bins) {
    features.clear();
    
    //create 3D histogram
    //total bins = bins^3 (ex: 8x8x8 = 512 bins)
    int total_bins = bins * bins * bins;
    std::vector<int> histogram(total_bins, 0);
    
    //calculate bin size (256 / bins)
    int bin_size = 256 / bins;
    
    //iterate through all pixels and populate histogram
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            
            //get bin indices for B, G, R
            int b_bin = pixel[0] / bin_size;
            int g_bin = pixel[1] / bin_size;
            int r_bin = pixel[2] / bin_size;
            
            //handle edge case where pixel value is 255
            if (b_bin >= bins) b_bin = bins - 1;
            if (g_bin >= bins) g_bin = bins - 1;
            if (r_bin >= bins) r_bin = bins - 1;
            
            //calculate 1D index from 3D indices
            int index = r_bin * bins * bins + g_bin * bins + b_bin;
            histogram[index]++;
        }
    }
    
    //convert to float vector 
    int total_pixels = src.rows * src.cols;
    for (int i = 0; i < total_bins; i++) {
        features.push_back(static_cast<float>(histogram[i]) / total_pixels);
    }
    
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

//histogram intersection distance metric
float histogram_intersection(const std::vector<float> &h1, const std::vector<float> &h2) {
    if (h1.size() != h2.size()) {
        std::cerr << "Error: Histograms must be same size" << std::endl;
        return -1.0f;
    }
    
    float intersection = 0.0f;
    
    //calculate intersection - sum of minimums
    for (size_t i = 0; i < h1.size(); i++) {
        intersection += std::min(h1[i], h2[i]);
    }
    
    //return as distance (1 - similarity)
    //intersection ranges from 0 to 1, so distance is 1 - intersection
    return 1.0f - intersection;
}

//Task 3: Extract multi-histogram (top and bottom halves)
int multi_histogram_features(cv::Mat &src, std::vector<float> &features, int bins) {
    features.clear();
    
    int total_bins = bins * bins * bins;
    int bin_size = 256 / bins;
    
    //split image into top and bottom halves
    int mid_row = src.rows / 2;
    
    //top half histogram
    std::vector<int> top_histogram(total_bins, 0);
    int top_pixels = 0;
    
    for (int i = 0; i < mid_row; i++) {
        for (int j = 0; j < src.cols; j++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            
            int b_bin = pixel[0] / bin_size;
            int g_bin = pixel[1] / bin_size;
            int r_bin = pixel[2] / bin_size;
            
            if (b_bin >= bins) b_bin = bins - 1;
            if (g_bin >= bins) g_bin = bins - 1;
            if (r_bin >= bins) r_bin = bins - 1;
            
            int index = r_bin * bins * bins + g_bin * bins + b_bin;
            top_histogram[index]++;
            top_pixels++;
        }
    }
    
    //bottom half histogram
    std::vector<int> bottom_histogram(total_bins, 0);
    int bottom_pixels = 0;
    
    for (int i = mid_row; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            
            int b_bin = pixel[0] / bin_size;
            int g_bin = pixel[1] / bin_size;
            int r_bin = pixel[2] / bin_size;
            
            if (b_bin >= bins) b_bin = bins - 1;
            if (g_bin >= bins) g_bin = bins - 1;
            if (r_bin >= bins) r_bin = bins - 1;
            
            int index = r_bin * bins * bins + g_bin * bins + b_bin;
            bottom_histogram[index]++;
            bottom_pixels++;
        }
    }
    
    //normalize and store top half histogram
    for (int i = 0; i < total_bins; i++) {
        features.push_back(static_cast<float>(top_histogram[i]) / top_pixels);
    }
    
    //normalize and store bottom half histogram
    for (int i = 0; i < total_bins; i++) {
        features.push_back(static_cast<float>(bottom_histogram[i]) / bottom_pixels);
    }
    
    //total features: 2 * bins^3 (ex: 2 * 512 = 1024 for 8 bins)
    return 0;
}

//multi-histogram distance with weighted combination
float multi_histogram_distance(const std::vector<float> &f1, const std::vector<float> &f2) {
    if (f1.size() != f2.size()) {
        std::cerr << "Error: Feature vectors must be same size" << std::endl;
        return -1.0f;
    }
    
    //split features into top and bottom histograms
    int histogram_size = f1.size() / 2;
    
    //extract top half histograms
    std::vector<float> f1_top(f1.begin(), f1.begin() + histogram_size);
    std::vector<float> f2_top(f2.begin(), f2.begin() + histogram_size);
    
    //extract bottom half histograms
    std::vector<float> f1_bottom(f1.begin() + histogram_size, f1.end());
    std::vector<float> f2_bottom(f2.begin() + histogram_size, f2.end());
    
    //compute histogram intersection for each half
    float top_distance = histogram_intersection(f1_top, f2_top);
    float bottom_distance = histogram_intersection(f1_bottom, f2_bottom);
    
    //equal weighting - 0.5 each
    float weighted_distance = 0.5 * top_distance + 0.5 * bottom_distance;
    
    return weighted_distance;
}

//Task 4: Extract texture features using Sobel gradient magnitude
int texture_features(cv::Mat &src, std::vector<float> &features, int bins) {
    features.clear();
    
    //convert to grayscale
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }
    
    //compute Sobel gradients
    cv::Mat grad_x, grad_y;
    cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);
    
    //compute gradient magnitude
    cv::Mat magnitude;
    cv::magnitude(grad_x, grad_y, magnitude);
    
    //create histogram of gradient magnitudes
    std::vector<int> histogram(bins, 0);
    
    //find max magnitude for normalization
    double max_mag;
    cv::minMaxLoc(magnitude, nullptr, &max_mag);
    
    if (max_mag == 0) max_mag = 1.0; //avoid division by zero
    
    //populate histogram
    for (int i = 0; i < magnitude.rows; i++) {
        for (int j = 0; j < magnitude.cols; j++) {
            float mag = magnitude.at<float>(i, j);
            int bin = static_cast<int>((mag / max_mag) * bins);
            if (bin >= bins) bin = bins - 1;
            histogram[bin]++;
        }
    }
    
    //normalize and convert to float
    int total_pixels = magnitude.rows * magnitude.cols;
    for (int i = 0; i < bins; i++) {
        features.push_back(static_cast<float>(histogram[i]) / total_pixels);
    }
    
    return 0;
}

//Task 4: Extract combined texture and color features
int texture_color_features(cv::Mat &src, std::vector<float> &features, int color_bins, int texture_bins) {
    features.clear();
    
    //extract color histogram
    std::vector<float> color_features;
    if (histogram_features(src, color_features, color_bins) != 0) {
        return -1;
    }
    
    //extract texture histogram
    std::vector<float> texture_feats;
    if (texture_features(src, texture_feats, texture_bins) != 0) {
        return -1;
    }
    
    //combine color features then texture features
    features.insert(features.end(), color_features.begin(), color_features.end());
    features.insert(features.end(), texture_feats.begin(), texture_feats.end());
    
    //total features: color_bins^3 + texture_bins
    return 0;
}

//texture + color distance with equal weighting
float texture_color_distance(const std::vector<float> &f1, const std::vector<float> &f2, int color_bins, int texture_bins) {
    if (f1.size() != f2.size()) {
        std::cerr << "Error: Feature vectors must be same size" << std::endl;
        return -1.0f;
    }
    
    int color_size = color_bins * color_bins * color_bins;
    int texture_size = texture_bins;
    
    //extract color histograms
    std::vector<float> f1_color(f1.begin(), f1.begin() + color_size);
    std::vector<float> f2_color(f2.begin(), f2.begin() + color_size);
    
    //extract texture histograms
    std::vector<float> f1_texture(f1.begin() + color_size, f1.begin() + color_size + texture_size);
    std::vector<float> f2_texture(f2.begin() + color_size, f2.begin() + color_size + texture_size);
    
    //compute histogram intersection for each
    float color_distance = histogram_intersection(f1_color, f2_color);
    float texture_distance = histogram_intersection(f1_texture, f2_texture);
    
    //equal weighting - 0.5 each
    float weighted_distance = 0.5 * color_distance + 0.5 * texture_distance;
    
    return weighted_distance;
}