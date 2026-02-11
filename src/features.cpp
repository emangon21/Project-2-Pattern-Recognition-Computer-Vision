/*
  Feature extraction implementations
*/
#include "features.h"
#include <cmath>
#include <iostream>
#include <algorithm>

// -------------------- Task 1 --------------------
// Extract 7x7 center square as baseline feature
int baseline_features(cv::Mat &src, std::vector<float> &features) {
    features.clear();

    int center_row = src.rows / 2;
    int center_col = src.cols / 2;
    int half_size = 3;

    if (center_row - half_size < 0 || center_row + half_size >= src.rows ||
        center_col - half_size < 0 || center_col + half_size >= src.cols) {
        std::cerr << "Error: Image too small for 7x7 center extraction" << std::endl;
        return -1;
    }

    for (int i = -half_size; i <= half_size; i++) {
        for (int j = -half_size; j <= half_size; j++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(center_row + i, center_col + j);
            features.push_back((float)pixel[0]);
            features.push_back((float)pixel[1]);
            features.push_back((float)pixel[2]);
        }
    }

    return 0;
}

// SSD distance metric
float ssd_distance(const std::vector<float> &f1, const std::vector<float> &f2) {
    if (f1.size() != f2.size()) {
        std::cerr << "Error: Feature vectors must be same size" << std::endl;
        return -1.0f;
    }
    float sum = 0.0f;
    for (size_t i = 0; i < f1.size(); i++) {
        float d = f1[i] - f2[i];
        sum += d * d;
    }
    return sum;
}

// -------------------- Task 2 --------------------
// Extract RGB color histogram
int histogram_features(cv::Mat &src, std::vector<float> &features, int bins) {
    features.clear();

    int total_bins = bins * bins * bins;
    std::vector<int> histogram(total_bins, 0);

    int bin_size = 256 / bins;

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);

            int b_bin = pixel[0] / bin_size;
            int g_bin = pixel[1] / bin_size;
            int r_bin = pixel[2] / bin_size;

            if (b_bin >= bins) b_bin = bins - 1;
            if (g_bin >= bins) g_bin = bins - 1;
            if (r_bin >= bins) r_bin = bins - 1;

            int index = r_bin * bins * bins + g_bin * bins + b_bin;
            histogram[index]++;
        }
    }

    int total_pixels = src.rows * src.cols;
    for (int i = 0; i < total_bins; i++) {
        features.push_back((float)histogram[i] / (float)total_pixels);
    }

    return 0;
}

// Histogram intersection distance metric
float histogram_intersection(const std::vector<float> &h1, const std::vector<float> &h2) {
    if (h1.size() != h2.size()) {
        std::cerr << "Error: Histograms must be same size" << std::endl;
        return -1.0f;
    }
    float inter = 0.0f;
    for (size_t i = 0; i < h1.size(); i++) inter += std::min(h1[i], h2[i]);
    return 1.0f - inter;
}

// -------------------- Task 3 --------------------
// Extract multi-histogram (top and bottom halves)
int multi_histogram_features(cv::Mat &src, std::vector<float> &features, int bins) {
    features.clear();

    int total_bins = bins * bins * bins;
    int bin_size = 256 / bins;
    int mid_row = src.rows / 2;

    std::vector<int> top_hist(total_bins, 0);
    std::vector<int> bot_hist(total_bins, 0);
    int top_pixels = 0, bot_pixels = 0;

    for (int i = 0; i < mid_row; i++) {
        for (int j = 0; j < src.cols; j++) {
            cv::Vec3b p = src.at<cv::Vec3b>(i, j);
            int b = p[0] / bin_size, g = p[1] / bin_size, r = p[2] / bin_size;
            if (b >= bins) b = bins-1; if (g >= bins) g = bins-1; if (r >= bins) r = bins-1;
            top_hist[r*bins*bins + g*bins + b]++; top_pixels++;
        }
    }
    for (int i = mid_row; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            cv::Vec3b p = src.at<cv::Vec3b>(i, j);
            int b = p[0] / bin_size, g = p[1] / bin_size, r = p[2] / bin_size;
            if (b >= bins) b = bins-1; if (g >= bins) g = bins-1; if (r >= bins) r = bins-1;
            bot_hist[r*bins*bins + g*bins + b]++; bot_pixels++;
        }
    }

    if (top_pixels <= 0) top_pixels = 1;
    if (bot_pixels <= 0) bot_pixels = 1;

    for (int i = 0; i < total_bins; i++) features.push_back((float)top_hist[i] / (float)top_pixels);
    for (int i = 0; i < total_bins; i++) features.push_back((float)bot_hist[i] / (float)bot_pixels);

    return 0;
}

// Multi-histogram distance
float multi_histogram_distance(const std::vector<float> &f1, const std::vector<float> &f2) {
    if (f1.size() != f2.size()) {
        std::cerr << "Error: Feature vectors must be same size" << std::endl;
        return -1.0f;
    }
    int half = (int)f1.size() / 2;
    std::vector<float> a1(f1.begin(), f1.begin()+half);
    std::vector<float> b1(f2.begin(), f2.begin()+half);
    std::vector<float> a2(f1.begin()+half, f1.end());
    std::vector<float> b2(f2.begin()+half, f2.end());
    return 0.5f*histogram_intersection(a1,b1) + 0.5f*histogram_intersection(a2,b2);
}

// -------------------- Task 4 --------------------
// Extract texture features using Sobel gradient magnitude
int texture_features(cv::Mat &src, std::vector<float> &features, int bins) {
    features.clear();

    cv::Mat gray;
    if (src.channels() == 3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else gray = src.clone();

    cv::Mat gx, gy, mag;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, mag);

    std::vector<int> hist(bins, 0);
    double max_mag = 0.0;
    cv::minMaxLoc(mag, nullptr, &max_mag);
    if (max_mag <= 0.0) max_mag = 1.0;

    for (int r = 0; r < mag.rows; r++) {
        const float *p = mag.ptr<float>(r);
        for (int c = 0; c < mag.cols; c++) {
            int b = (int)((p[c] / (float)max_mag) * bins);
            if (b >= bins) b = bins - 1;
            hist[b]++;
        }
    }

    int total = mag.rows * mag.cols;
    if (total <= 0) total = 1;
    for (int i = 0; i < bins; i++) features.push_back((float)hist[i] / (float)total);

    return 0;
}

// Extract combined texture + color features
int texture_color_features(cv::Mat &src, std::vector<float> &features,
                           int color_bins, int texture_bins) {
    features.clear();
    std::vector<float> c, t;
    if (histogram_features(src, c, color_bins) != 0) return -1;
    if (texture_features(src, t, texture_bins) != 0) return -1;
    features.insert(features.end(), c.begin(), c.end());
    features.insert(features.end(), t.begin(), t.end());
    return 0;
}

// Texture + color distance
float texture_color_distance(const std::vector<float> &f1, const std::vector<float> &f2,
                             int color_bins, int texture_bins) {
    if (f1.size() != f2.size()) {
        std::cerr << "Error: Feature vectors must be same size" << std::endl;
        return -1.0f;
    }

    int color_size = color_bins * color_bins * color_bins;
    int tex_size = texture_bins;

    std::vector<float> aC(f1.begin(), f1.begin()+color_size);
    std::vector<float> bC(f2.begin(), f2.begin()+color_size);

    std::vector<float> aT(f1.begin()+color_size, f1.begin()+color_size+tex_size);
    std::vector<float> bT(f2.begin()+color_size, f2.begin()+color_size+tex_size);

    return 0.5f*histogram_intersection(aC,bC) + 0.5f*histogram_intersection(aT,bT);
}

// -------------------- Task 5 --------------------
// Cosine distance for 512-D embeddings
float cosine_distance(const std::vector<float> &v1, const std::vector<float> &v2) {
    if (v1.size() != v2.size()) {
        std::cerr << "Error: vectors must be the same size\n";
        return -1.0f;
    }
    double dot=0.0, n1=0.0, n2=0.0;
    for (size_t i=0;i<v1.size();i++){
        dot += (double)v1[i]*(double)v2[i];
        n1  += (double)v1[i]*(double)v1[i];
        n2  += (double)v2[i]*(double)v2[i];
    }
    if (n1==0.0 || n2==0.0) return 1.0f;
    return (float)(1.0 - dot/(std::sqrt(n1)*std::sqrt(n2)));
}

// -------------------- Task 7 --------------------
float edge_density(const cv::Mat &src) {
    cv::Mat gray;
    if (src.channels() == 3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else gray = src.clone();

    cv::Mat sx, sy, mag;
    cv::Sobel(gray, sx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, sy, CV_32F, 0, 1, 3);
    cv::magnitude(sx, sy, mag);

    const float thresh = 50.0f;
    int strong = 0;
    int total = mag.rows * mag.cols;

    for (int r = 0; r < mag.rows; r++) {
        const float *p = mag.ptr<float>(r);
        for (int c = 0; c < mag.cols; c++) if (p[c] > thresh) strong++;
    }
    return (total > 0) ? (float)strong / (float)total : 0.0f;
}

int custom_hsv_edge_features(const cv::Mat &src, std::vector<float> &features,
                             int h_bins, int s_bins) {
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
            int H = p[c][0];
            int S = p[c][1];

            int hb = (H * h_bins) / 180; if (hb >= h_bins) hb = h_bins - 1;
            int sb = (S * s_bins) / 256; if (sb >= s_bins) sb = s_bins - 1;

            hist[hb * s_bins + sb] += 1.0f;
        }
    }

    float denom = (float)(top.rows * top.cols);
    if (denom > 0.0f) for (float &v : hist) v /= denom;

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
    for (size_t i = 0; i < a.size(); i++) inter += std::min((double)a[i], (double)b[i]);
    double d = 1.0 - inter;
    if (d < 0.0) d = 0.0;
    if (d > 1.0) d = 1.0;
    return (float)d;
}

float custom_distance(const std::vector<float> &target, const std::vector<float> &db,
                      int h_bins, int s_bins) {
    int hist_len = h_bins * s_bins;
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

// -------------------- Extensions --------------------
static inline float safe_div(float a, float b) { return (b <= 1e-6f) ? 0.0f : (a / b); }

// Banana feature vector extractor (9-D)
int banana_feature_vector(const cv::Mat &img, std::vector<float> &feat,
                          float hsv_s_min, float hsv_v_min) {
    feat.assign(9, 0.0f);
    if (img.empty()) return -1;

    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    // yellow-ish banana range (H fixed, S/V minimums controlled by args)
    cv::Mat mask;
    cv::inRange(hsv,
                cv::Scalar(18, (int)(hsv_s_min * 255.0f), (int)(hsv_v_min * 255.0f)),
                cv::Scalar(33, 255, 255),
                mask);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);

    int best = -1, bestArea = 0;
    for (int i = 1; i < n; i++) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > bestArea) { bestArea = area; best = i; }
    }

    if (best < 0 || bestArea < 50) {
        feat.assign(9, 0.0f);
        return 0;
    }

    cv::Mat blob = (labels == best);
    blob.convertTo(blob, CV_8U, 255);

    float img_area = (float)(img.rows * img.cols);
    float area_ratio = (img_area > 0) ? (float)bestArea / img_area : 0.0f;

    float cx = (float)centroids.at<double>(best, 0);
    float cy = (float)centroids.at<double>(best, 1);
    float cx_n = (img.cols > 0) ? cx / (float)img.cols : 0.0f;
    float cy_n = (img.rows > 0) ? cy / (float)img.rows : 0.0f;

    // var_x, var_y (normalized)
    double sx = 0, sy = 0, sxx = 0, syy = 0;
    int count = 0;
    for (int r = 0; r < blob.rows; r++) {
        const uchar *p = blob.ptr<uchar>(r);
        for (int c = 0; c < blob.cols; c++) {
            if (p[c]) {
                sx += c; sy += r;
                sxx += (double)c * c;
                syy += (double)r * r;
                count++;
            }
        }
    }

    double meanx = (count > 0) ? sx / count : 0.0;
    double meany = (count > 0) ? sy / count : 0.0;
    double var_x = (count > 0) ? (sxx / count - meanx * meanx) : 0.0;
    double var_y = (count > 0) ? (syy / count - meany * meany) : 0.0;

    if (img.cols > 0) var_x /= (double)(img.cols * img.cols);
    if (img.rows > 0) var_y /= (double)(img.rows * img.rows);

    // bbox/extent/solidity
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(blob, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    float bbox_aspect = 0.0f, extent = 0.0f, solidity = 0.0f;
    if (!contours.empty()) {
        int bi = 0;
        double bestC = 0;
        for (int i = 0; i < (int)contours.size(); i++) {
            double a = cv::contourArea(contours[i]);
            if (a > bestC) { bestC = a; bi = i; }
        }

        cv::Rect bbox = cv::boundingRect(contours[bi]);
        double bboxArea = (double)bbox.width * (double)bbox.height;

        if (bbox.width > 0 && bbox.height > 0) {
            bbox_aspect = (float)std::max(bbox.width, bbox.height) / (float)std::min(bbox.width, bbox.height);
        }
        if (bboxArea > 0) extent = (float)(bestC / bboxArea);

        std::vector<cv::Point> hull;
        cv::convexHull(contours[bi], hull);
        double hullArea = cv::contourArea(hull);
        if (hullArea > 0) solidity = (float)(bestC / hullArea);
    }

    // edge density inside blob
    cv::Mat gray, edges;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, edges, 80, 160);
    cv::Mat edges_in_blob;
    cv::bitwise_and(edges, blob, edges_in_blob);
    float ed = (float)cv::countNonZero(edges_in_blob) / (float)std::max(1, cv::countNonZero(blob));

    feat[0] = area_ratio;
    feat[1] = cx_n;
    feat[2] = cy_n;
    feat[3] = (float)var_x;
    feat[4] = (float)var_y;
    feat[5] = ed;
    feat[6] = bbox_aspect;
    feat[7] = extent;
    feat[8] = solidity;

    return 0;
}

// Banana distance
float banana_distance(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != 9 || b.size() != 9) return 1e9f;

    const float AREA_MIN = 0.0025f;
    const float AREA_MAX = 0.25f;
    bool a_has = (a[0] >= AREA_MIN && a[0] <= AREA_MAX);
    bool b_has = (b[0] >= AREA_MIN && b[0] <= AREA_MAX);
    if (!a_has || !b_has) return 1e6f;

    const float ASPECT_MIN = 1.7f;
    const float EXTENT_MIN = 0.18f;
    const float SOLID_MIN  = 0.55f;
    bool a_ok = (a[6] >= ASPECT_MIN) && (a[7] >= EXTENT_MIN) && (a[8] >= SOLID_MIN);
    bool b_ok = (b[6] >= ASPECT_MIN) && (b[7] >= EXTENT_MIN) && (b[8] >= SOLID_MIN);
    if (!a_ok || !b_ok) return 1e6f;

    float d = 0.0f;
    d += 2.0f * std::fabs(a[0] - b[0]);
    d += 0.2f * std::fabs(a[1] - b[1]);
    d += 0.2f * std::fabs(a[2] - b[2]);

    d += 7.0f * std::fabs(a[3] - b[3]);
    d += 7.0f * std::fabs(a[4] - b[4]);

    d += 0.6f * std::fabs(a[5] - b[5]);

    d += 1.0f * std::fabs(a[6] - b[6]);
    d += 0.8f * std::fabs(a[7] - b[7]);
    d += 0.8f * std::fabs(a[8] - b[8]);

    return d;
}

// Blue bin distance
float bluebin_distance(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != 9 || b.size() != 9) return 1e9f;

    float d = 0.0f;
    d += 4.0f * std::fabs(a[0] - b[0]);
    d += 1.0f * std::fabs(a[1] - b[1]);
    d += 1.0f * std::fabs(a[2] - b[2]);
    d += 2.0f * std::fabs(a[3] - b[3]);
    d += 2.0f * std::fabs(a[4] - b[4]);
    d += 0.5f * std::fabs(a[5] - b[5]);

    d += 2.0f * std::fabs(a[6] - b[6]);
    d += 1.5f * std::fabs(a[7] - b[7]);
    d += 1.5f * std::fabs(a[8] - b[8]);

    const float AREA_MIN = 0.004f;
    const float AREA_MAX = 0.25f;
    bool a_ok = (a[0] >= AREA_MIN && a[0] <= AREA_MAX);
    bool b_ok = (b[0] >= AREA_MIN && b[0] <= AREA_MAX);
    if (a_ok != b_ok) d += 2.0f;

    return d;
}

// Face metric distance
float face_metric_distance(const std::vector<float> &a, const std::vector<float> &b) {
    const int stats_len = 5;
    const int hist_len  = 512;
    const int emb_len   = 512;
    const int expected  = stats_len + hist_len + emb_len;
    if ((int)a.size() != expected || (int)b.size() != expected) return 1e9f;

    float a_has = a[0];
    float b_has = b[0];

    if (a_has >= 0.5f && b_has < 0.5f) return 5.0f;

    if (a_has < 0.5f) {
        std::vector<float> a_emb(a.begin() + stats_len + hist_len, a.end());
        std::vector<float> b_emb(b.begin() + stats_len + hist_len, b.end());
        return cosine_distance(a_emb, b_emb);
    }

    std::vector<float> a_hist(a.begin() + stats_len, a.begin() + stats_len + hist_len);
    std::vector<float> b_hist(b.begin() + stats_len, b.begin() + stats_len + hist_len);

    float D_face = histogram_intersection(a_hist, b_hist);

    std::vector<float> a_emb(a.begin() + stats_len + hist_len, a.end());
    std::vector<float> b_emb(b.begin() + stats_len + hist_len, b.end());
    float D_emb = cosine_distance(a_emb, b_emb);

    float geom = 0.0f;
    geom += std::fabs(a[1] - b[1]);
    geom += std::fabs(a[2] - b[2]);
    geom += std::fabs(a[3] - b[3]);
    geom += 0.2f * std::fabs(a[4] - b[4]);

    return 0.65f * D_face + 0.30f * D_emb + 0.05f * geom;
}

// -------------------- Method A: HSV Spatial Moments --------------------
int hsv_spatial_moments(const cv::Mat &src, std::vector<float> &feat, int grid) {
    if (src.empty() || grid <= 0) return -1;

    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    feat.clear();
    feat.reserve(grid * grid * 6);

    int cell_w = hsv.cols / grid;
    int cell_h = hsv.rows / grid;

    for (int gy = 0; gy < grid; gy++) {
        for (int gx = 0; gx < grid; gx++) {
            int x0 = gx * cell_w;
            int y0 = gy * cell_h;
            int x1 = (gx == grid-1) ? hsv.cols : (gx+1) * cell_w;
            int y1 = (gy == grid-1) ? hsv.rows : (gy+1) * cell_h;

            cv::Rect roi(x0, y0, x1 - x0, y1 - y0);
            cv::Mat cell = hsv(roi);

            cv::Scalar mean, stddev;
            cv::meanStdDev(cell, mean, stddev);

            feat.push_back((float)(mean[0] / 179.0));
            feat.push_back((float)(stddev[0] / 179.0));
            feat.push_back((float)(mean[1] / 255.0));
            feat.push_back((float)(stddev[1] / 255.0));
            feat.push_back((float)(mean[2] / 255.0));
            feat.push_back((float)(stddev[2] / 255.0));
        }
    }
    return 0;
}

float l2_distance(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size()) return 1e9f;
    double s = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double d = (double)a[i] - (double)b[i];
        s += d*d;
    }
    return (float)std::sqrt(s);
}

// -------------------- Method B: LBP Texture Histogram (single copy!) --------------------
static inline unsigned char lbp_code_8(const cv::Mat &g, int y, int x) {
    unsigned char c = g.at<unsigned char>(y, x);
    unsigned char code = 0;
    code |= (g.at<unsigned char>(y-1, x-1) >= c) << 7;
    code |= (g.at<unsigned char>(y-1, x  ) >= c) << 6;
    code |= (g.at<unsigned char>(y-1, x+1) >= c) << 5;
    code |= (g.at<unsigned char>(y,   x+1) >= c) << 4;
    code |= (g.at<unsigned char>(y+1, x+1) >= c) << 3;
    code |= (g.at<unsigned char>(y+1, x  ) >= c) << 2;
    code |= (g.at<unsigned char>(y+1, x-1) >= c) << 1;
    code |= (g.at<unsigned char>(y,   x-1) >= c) << 0;
    return code;
}

int lbp_histogram_features(const cv::Mat &src, std::vector<float> &features) {
    if (src.empty()) return -1;

    cv::Mat gray;
    if (src.channels() == 3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else gray = src.clone();

    std::vector<int> hist(256, 0);
    int count = 0;

    for (int y = 1; y < gray.rows - 1; y++) {
        for (int x = 1; x < gray.cols - 1; x++) {
            unsigned char code = lbp_code_8(gray, y, x);
            hist[(int)code]++;
            count++;
        }
    }

    if (count <= 0) count = 1;
    features.clear();
    features.reserve(256);
    for (int i = 0; i < 256; i++) features.push_back((float)hist[i] / (float)count);

    return 0;
}

float chi_square_distance(const std::vector<float> &h1, const std::vector<float> &h2) {
    if (h1.size() != h2.size()) return 1e9f;

    const double eps = 1e-10;
    double sum = 0.0;
    for (size_t i = 0; i < h1.size(); i++) {
        double a = (double)h1[i];
        double b = (double)h2[i];
        double num = a - b;
        double den = a + b + eps;
        sum += (num * num) / den;
    }
    return (float)(0.5 * sum);
}
