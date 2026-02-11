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

// Task 5
float cosine_distance(const std::vector<float> &v1, const std::vector<float> &v2) {
    if (v1.size() != v2.size()) {
        std::cerr << "Error: vectors must be the same size\n";
        return -1.0f;
    }

    double dot = 0.0;
    double n1 = 0.0;
    double n2 = 0.0;

    for (size_t i = 0; i < v1.size(); i++) {
        dot += (double)v1[i] * (double)v2[i];
        n1  += (double)v1[i] * (double)v1[i];
        n2  += (double)v2[i] * (double)v2[i];
    }

    if (n1 == 0.0 || n2 == 0.0) return 1.0f; // max distance if a vector is zero

    double cos_sim = dot / (std::sqrt(n1) * std::sqrt(n2));
    return (float)(1.0 - cos_sim); // cosine distance
}

// Task 7

#include <cmath>
#include <algorithm>

float edge_density(const cv::Mat &src) {
    cv::Mat gray;
    if (src.channels() == 3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else gray = src.clone();

    cv::Mat sx, sy;
    cv::Sobel(gray, sx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, sy, CV_32F, 0, 1, 3);

    cv::Mat mag;
    cv::magnitude(sx, sy, mag);

    const float thresh = 50.0f;
    int strong = 0;
    int total = mag.rows * mag.cols;

    for (int r = 0; r < mag.rows; r++) {
        const float *p = mag.ptr<float>(r);
        for (int c = 0; c < mag.cols; c++) {
            if (p[c] > thresh) strong++;
        }
    }
    return (total > 0) ? (float)strong / (float)total : 0.0f;
}

int custom_hsv_edge_features(const cv::Mat &src, std::vector<float> &features, int h_bins, int s_bins) {
    if (src.empty()) return -1;
    if (h_bins <= 0 || s_bins <= 0) return -1;

    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    int half = hsv.rows / 2;
    cv::Rect roi(0, 0, hsv.cols, half);
    cv::Mat top = hsv(roi);

    std::vector<float> hist(h_bins * s_bins, 0.0f);

    for (int r = 0; r < top.rows; r++) {
        const cv::Vec3b *p = top.ptr<cv::Vec3b>(r);
        for (int c = 0; c < top.cols; c++) {
            int H = p[c][0]; // 0..179
            int S = p[c][1]; // 0..255

            int hb = (H * h_bins) / 180;
            if (hb >= h_bins) hb = h_bins - 1;

            int sb = (S * s_bins) / 256;
            if (sb >= s_bins) sb = s_bins - 1;

            hist[hb * s_bins + sb] += 1.0f;
        }
    }

    float denom = (float)(top.rows * top.cols);
    if (denom > 0.0f) {
        for (float &v : hist) v /= denom;
    }

    float ed = edge_density(src);

    features.clear();
    features.reserve(hist.size() + 1);
    features.insert(features.end(), hist.begin(), hist.end());
    features.push_back(ed);

    return 0;
}

static float hist_intersection_distance_custom(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size()) return 1.0f;
    double inter = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        inter += std::min((double)a[i], (double)b[i]);
    }
    double d = 1.0 - inter;
    if (d < 0.0) d = 0.0;
    if (d > 1.0) d = 1.0;
    return (float)d;
}

float custom_distance(const std::vector<float> &target, const std::vector<float> &db, int h_bins, int s_bins) {
    int hist_len = h_bins * s_bins;    // 108 if 18x6
    int edge_len = 1;
    int emb_len  = 512;

    int expected = hist_len + edge_len + emb_len;
    if ((int)target.size() != expected || (int)db.size() != expected) return 1e9f;

    std::vector<float> t_hist(target.begin(), target.begin() + hist_len);
    std::vector<float> d_hist(db.begin(), db.begin() + hist_len);

    float t_edge = target[hist_len];
    float d_edge = db[hist_len];

    std::vector<float> t_emb(target.begin() + hist_len + edge_len, target.end());
    std::vector<float> d_emb(db.begin() + hist_len + edge_len, db.end());

    float D_hsv  = hist_intersection_distance_custom(t_hist, d_hist);
    float D_edge = std::fabs(t_edge - d_edge);
    float D_emb  = cosine_distance(t_emb, d_emb);

    return 0.45f * D_hsv + 0.10f * D_edge + 0.45f * D_emb;
}

// Extension: Banana Color Blob

float banana_distance(const std::vector<float> &a,
                      const std::vector<float> &b) {
    if (a.size() != 9 || b.size() != 9) return 1e9f;

    float d = 0.0f;

    // base distance terms
    d += 4.0f * std::fabs(a[0] - b[0]); // area_ratio
    d += 1.0f * std::fabs(a[1] - b[1]); // cx
    d += 1.0f * std::fabs(a[2] - b[2]); // cy
    d += 2.0f * std::fabs(a[3] - b[3]); // var_x
    d += 2.0f * std::fabs(a[4] - b[4]); // var_y
    d += 0.5f * std::fabs(a[5] - b[5]); // edge_density

    // shape terms
    d += 3.0f * std::fabs(a[6] - b[6]); // bbox_aspect
    d += 2.0f * std::fabs(a[7] - b[7]); // extent
    d += 2.0f * std::fabs(a[8] - b[8]); // solidity

    // ---- HARD PENALTIES (big improvement) ----
    // If one is elongated like a banana and the other isn't, push it down.
    const float ASPECT_MIN = 1.8f;      // banana-ish elongation
    const float EXTENT_MIN = 0.20f;     // avoid tiny thin noise blobs
    const float SOLID_MIN  = 0.65f;     // avoid very jagged regions

    bool a_ok = (a[6] >= ASPECT_MIN) && (a[7] >= EXTENT_MIN) && (a[8] >= SOLID_MIN);
    bool b_ok = (b[6] >= ASPECT_MIN) && (b[7] >= EXTENT_MIN) && (b[8] >= SOLID_MIN);

    if (a_ok != b_ok) d += 2.5f;        // strong penalty for mismatch
    if (a[0] > 0.30f || b[0] > 0.30f) d += 1.0f;  // likely yellow wall/sign (too much yellow)

    return d;
}

float bluebin_distance(const std::vector<float> &a,
                       const std::vector<float> &b) {
    if (a.size() != 9 || b.size() != 9) return 1e9f;

    float d = 0.0f;

    d += 4.0f * std::fabs(a[0] - b[0]); // area_ratio
    d += 1.0f * std::fabs(a[1] - b[1]); // cx
    d += 1.0f * std::fabs(a[2] - b[2]); // cy
    d += 2.0f * std::fabs(a[3] - b[3]); // var_x
    d += 2.0f * std::fabs(a[4] - b[4]); // var_y
    d += 0.5f * std::fabs(a[5] - b[5]); // edge_density

    d += 2.0f * std::fabs(a[6] - b[6]); // bbox_aspect
    d += 1.5f * std::fabs(a[7] - b[7]); // extent
    d += 1.5f * std::fabs(a[8] - b[8]); // solidity

    // ---- hard penalties to suppress sky/water/blue walls ----
    const float AREA_MIN = 0.004f;   // too small = noise
    const float AREA_MAX = 0.25f;    // too large = sky/wall
    const float EXT_MIN  = 0.18f;    // filter thin scattered blobs
    const float SOL_MIN  = 0.55f;

    bool a_ok = (a[0] >= AREA_MIN && a[0] <= AREA_MAX && a[7] >= EXT_MIN && a[8] >= SOL_MIN);
    bool b_ok = (b[0] >= AREA_MIN && b[0] <= AREA_MAX && b[7] >= EXT_MIN && b[8] >= SOL_MIN);

    if (a_ok != b_ok) d += 2.0f;

    return d;
}

float face_metric_distance(const std::vector<float> &a,
                           const std::vector<float> &b) {
    // Feature layout:
    // [has_face, area_ratio, cx, cy, aspect] + face_hist(512) + emb(512)
    const int stats_len = 5;
    const int hist_len  = 512;
    const int emb_len   = 512;
    const int expected  = stats_len + hist_len + emb_len; // 1029

    if ((int)a.size() != expected || (int)b.size() != expected) return 1e9f;

    float a_has = a[0];
    float b_has = b[0];

    // If target has face, strongly penalize db images without face
    if (a_has >= 0.5f && b_has < 0.5f) return 5.0f;

    // If target has no face, fall back to embedding distance only
    if (a_has < 0.5f) {
        std::vector<float> a_emb(a.begin() + stats_len + hist_len, a.end());
        std::vector<float> b_emb(b.begin() + stats_len + hist_len, b.end());
        return cosine_distance(a_emb, b_emb);
    }

    // Target HAS face and db HAS face → compare face hist + embedding
    std::vector<float> a_hist(a.begin() + stats_len, a.begin() + stats_len + hist_len);
    std::vector<float> b_hist(b.begin() + stats_len, b.begin() + stats_len + hist_len);

    float D_face = histogram_intersection(a_hist, b_hist); // 0..1 distance

    std::vector<float> a_emb(a.begin() + stats_len + hist_len, a.end());
    std::vector<float> b_emb(b.begin() + stats_len + hist_len, b.end());
    float D_emb = cosine_distance(a_emb, b_emb);

    // Small penalty if face geometry is very different
    float geom = 0.0f;
    geom += std::fabs(a[1] - b[1]); // area_ratio
    geom += std::fabs(a[2] - b[2]); // cx
    geom += std::fabs(a[3] - b[3]); // cy
    geom += 0.2f * std::fabs(a[4] - b[4]); // aspect (less important)

    // Weighted combo
    return 0.65f * D_face + 0.30f * D_emb + 0.05f * geom;
}


