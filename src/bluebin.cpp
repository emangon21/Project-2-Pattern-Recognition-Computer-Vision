#include <opencv2/opencv.hpp>
#include <cstdio>
#include <vector>
#include <string>
#include <filesystem>

#include "features.h"
#include "csv_util.h"

static std::string basename_only(const std::string &path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return path;
    return path.substr(pos + 1);
}

// 9D feature vector for blue bin:
// [area_ratio, cx, cy, var_x, var_y, edge_density, bbox_aspect, extent, solidity]
static int bluebin_features_db(const cv::Mat &img, std::vector<float> &feat) {
    if (img.empty()) return -1;

    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    // OpenCV Hue range is [0..179]
    // Blue is roughly ~90..130, tighten if needed
    cv::Mat mask;
    cv::inRange(hsv, cv::Scalar(90, 80, 60), cv::Scalar(130, 255, 255), mask);

    // clean mask
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    // keep largest connected component
    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);

    int best = -1;
    int bestArea = 0;
    for (int i = 1; i < n; i++) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > bestArea) { bestArea = area; best = i; }
    }

    float area_ratio = (float)bestArea / (float)(img.rows * img.cols);
    if (best <= 0 || bestArea <= 0) {
        feat = {0,0,0,0,0,0, 0,0,0};
        return 0;
    }

    cv::Mat blob = (labels == best);
    blob.convertTo(blob, CV_8U, 255);

    // centroid + variance
    cv::Moments m = cv::moments(blob, true);
    float cx = (m.m00 > 0) ? (float)(m.m10 / m.m00) : 0.0f;
    float cy = (m.m00 > 0) ? (float)(m.m01 / m.m00) : 0.0f;

    float cx_n = cx / (float)img.cols;
    float cy_n = cy / (float)img.rows;

    double var_x = 0.0, var_y = 0.0;
    {
        std::vector<cv::Point> pts;
        cv::findNonZero(blob, pts);
        if (!pts.empty()) {
            for (auto &p : pts) {
                double dx = p.x - cx;
                double dy = p.y - cy;
                var_x += dx * dx;
                var_y += dy * dy;
            }
            var_x /= pts.size();
            var_y /= pts.size();
            var_x /= (double)(img.cols * img.cols);
            var_y /= (double)(img.rows * img.rows);
        }
    }

    // shape features
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(blob, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    float bbox_aspect = 0.0f, extent = 0.0f, solidity = 0.0f;
    if (!contours.empty()) {
        int ci = 0;
        double bestA = 0.0;
        for (int i = 0; i < (int)contours.size(); i++) {
            double a = cv::contourArea(contours[i]);
            if (a > bestA) { bestA = a; ci = i; }
        }

        double area = cv::contourArea(contours[ci]);
        cv::Rect r = cv::boundingRect(contours[ci]);
        double bboxArea = (double)r.width * (double)r.height;

        if (r.width > 0 && r.height > 0) {
            double ar = (double)std::max(r.width, r.height) / (double)std::min(r.width, r.height);
            bbox_aspect = (float)ar;
        }
        if (bboxArea > 0) extent = (float)(area / bboxArea);

        std::vector<cv::Point> hull;
        cv::convexHull(contours[ci], hull);
        double hullArea = cv::contourArea(hull);
        if (hullArea > 0) solidity = (float)(area / hullArea);
    }

    float ed = edge_density(img);

    feat.clear();
    feat.reserve(9);
    feat.push_back(area_ratio);
    feat.push_back(cx_n);
    feat.push_back(cy_n);
    feat.push_back((float)var_x);
    feat.push_back((float)var_y);
    feat.push_back(ed);
    feat.push_back(bbox_aspect);
    feat.push_back(extent);
    feat.push_back(solidity);

    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <image_dir> <output_csv>\n", argv[0]);
        return -1;
    }

    char *image_dir = argv[1];
    char *out_csv   = argv[2];

    int reset_file = 1;
    int count = 0;

    for (const auto &entry : std::filesystem::directory_iterator(image_dir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".jpg") continue;

        cv::Mat img = cv::imread(entry.path().string());
        if (img.empty()) continue;

        std::vector<float> f;
        if (bluebin_features_db(img, f) != 0) continue;

        std::string base = basename_only(entry.path().string());
        append_image_data_csv(out_csv, (char*)base.c_str(), f, reset_file);
        reset_file = 0;
        count++;
    }

    printf("Wrote bluebin features for %d images\n", count);
    return 0;
}
