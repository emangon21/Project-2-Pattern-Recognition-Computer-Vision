/*
  Query program - Find top N similar images

  Usage: ./bin/query <target_image> <feature_csv> <method> <N> [bins]
*/
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>

#include "features.h"
#include "csv_util.h"

static std::string basename_only(const std::string &path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return path;
    return path.substr(pos + 1);
}

static cv::Mat load_db_image(const std::string &name, const std::string &img_dir) {
    cv::Mat img = cv::imread(name);
    if (!img.empty()) return img;

    std::string full = img_dir;
    if (!full.empty() && full.back() != '/') full += "/";
    full += name;
    return cv::imread(full);
}

static cv::Mat resize_or_blank(const cv::Mat &img, int w, int h) {
    if (img.empty()) {
        return cv::Mat(h, w, CV_8UC3, cv::Scalar(50, 50, 50));
    }
    cv::Mat r;
    cv::resize(img, r, cv::Size(w, h));
    return r;
}

static void draw_label(cv::Mat &img, const std::string &text) {
    cv::rectangle(img, cv::Point(0,0), cv::Point(img.cols, 22),
                  cv::Scalar(0,0,0), cv::FILLED);
    cv::putText(img, text, cv::Point(5,16),
                cv::FONT_HERSHEY_SIMPLEX, 0.45,
                cv::Scalar(0,255,255), 1);
}

struct ImageDistance {
    char *filename;
    float distance;
};

static bool compareDistance(const ImageDistance &a, const ImageDistance &b) {
    return a.distance < b.distance;
}

static float banana_edge_density(const cv::Mat &src) {
    cv::Mat gray;
    if (src.channels() == 3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    else gray = src.clone();

    cv::Mat sx, sy, mag;
    cv::Sobel(gray, sx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, sy, CV_32F, 0, 1, 3);
    cv::magnitude(sx, sy, mag);

    int strong = 0;
    int total = mag.rows * mag.cols;
    for (int r = 0; r < mag.rows; r++) {
        const float *p = mag.ptr<float>(r);
        for (int c = 0; c < mag.cols; c++) {
            if (p[c] > 50.0f) strong++;
        }
    }
    return (total > 0) ? (float)strong / (float)total : 0.0f;
}

static int banana_features_local(const cv::Mat &img, std::vector<float> &feat) {
    if (img.empty()) return -1;

    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    cv::Mat mask;
    cv::inRange(hsv, cv::Scalar(18, 100, 90), cv::Scalar(33, 255, 255), mask);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

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
        feat = {0,0,0,0,0,0,0,0,0};
        return 0;
    }

    cv::Mat blob = (labels == best);
    blob.convertTo(blob, CV_8U, 255);

    cv::Moments m = cv::moments(blob, true);
    float cx = (m.m00 > 0) ? (float)(m.m10 / m.m00) : 0.0f;
    float cy = (m.m00 > 0) ? (float)(m.m01 / m.m00) : 0.0f;

    float cx_n = cx / (float)img.cols;
    float cy_n = cy / (float)img.rows;

    double var_x = 0.0, var_y = 0.0;
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

    float ed = banana_edge_density(img);

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

static int bluebin_features_local(const cv::Mat &img, std::vector<float> &feat) {
    if (img.empty()) return -1;

    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    cv::Mat mask;
    cv::inRange(hsv, cv::Scalar(95, 110, 70), cv::Scalar(125, 255, 255), mask);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

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
        feat = {0,0,0,0,0,0,0,0,0};
        return 0;
    }

    cv::Mat blob = (labels == best);
    blob.convertTo(blob, CV_8U, 255);

    cv::Moments m = cv::moments(blob, true);
    float cx = (m.m00 > 0) ? (float)(m.m10 / m.m00) : 0.0f;
    float cy = (m.m00 > 0) ? (float)(m.m01 / m.m00) : 0.0f;

    float cx_n = cx / (float)img.cols;
    float cy_n = cy / (float)img.rows;

    double var_x = 0.0, var_y = 0.0;
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

static bool banana_present_local(const cv::Mat &img) {
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    cv::Mat mask;
    cv::inRange(hsv, cv::Scalar(18, 80, 80), cv::Scalar(33, 255, 255), mask);
    double ratio = (double)cv::countNonZero(mask) / (double)(img.rows * img.cols);
    return ratio > 0.05;
}

static bool bluebin_present_local(const cv::Mat &img) {
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    cv::Mat mask;
    cv::inRange(hsv, cv::Scalar(95, 80, 60), cv::Scalar(125, 255, 255), mask);
    double ratio = (double)cv::countNonZero(mask) / (double)(img.rows * img.cols);
    return ratio > 0.05;
}

static bool face_present_local(const cv::Mat &img) {
    std::string cascade =
        "/opt/homebrew/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";

    cv::CascadeClassifier cc;
    if (!cc.load(cascade)) return false;

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);

    std::vector<cv::Rect> faces;
    cc.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(40, 40));
    return !faces.empty();
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        printf("Usage: %s <target_image> <feature_csv> <method> <N> [bins]\n", argv[0]);
        return -1;
    }

    char *target_filename = argv[1];
    char *csv_filename = argv[2];
    std::string method = argv[3];
    int N = atoi(argv[4]);
    int bins = (argc >= 6) ? atoi(argv[5]) : 8;
    if (bins <= 0) bins = 8;

    cv::Mat target_image = cv::imread(target_filename);
    if (target_image.empty()) {
        printf("Error: Could not read target image %s\n", target_filename);
        return -1;
    }

    std::vector<char *> filenames;
    std::vector<std::vector<float>> database_features;

    if (read_image_data_csv(csv_filename, filenames, database_features, 0) != 0) {
        printf("Error: Could not read feature database\n");
        return -1;
    }

    if (method == "smart") {
        if (face_present_local(target_image)) method = "face";
        else if (banana_present_local(target_image)) method = "banana";
        else if (bluebin_present_local(target_image)) method = "bluebin";
        else method = "embedding";
        printf("Smart mode selected method: %s\n\n", method.c_str());
    }

    std::vector<float> target_features;
    std::vector<float> custom_part;

    if (method == "embedding" || method == "face") {
        std::string target_base = basename_only(std::string(target_filename));
        int target_idx = -1;
        for (size_t i = 0; i < filenames.size(); i++) {
            if (strcmp(filenames[i], target_filename) == 0 ||
                strcmp(filenames[i], target_base.c_str()) == 0) {
                target_idx = (int)i;
                break;
            }
        }
        if (target_idx < 0) {
            printf("Error: Could not find target in CSV\n");
            return -1;
        }
        target_features = database_features[target_idx];
    } else if (method == "baseline") {
        baseline_features(target_image, target_features);
    } else if (method == "histogram") {
        histogram_features(target_image, target_features, bins);
    } else if (method == "multi_histogram") {
        multi_histogram_features(target_image, target_features, bins);
    } else if (method == "texture_color") {
        texture_color_features(target_image, target_features, bins, 16);
    } else if (method == "banana") {
        banana_features_local(target_image, target_features);
    } else if (method == "bluebin") {
        bluebin_features_local(target_image, target_features);
    } else if (method == "custom") {
        int H_BINS = 18;
        int S_BINS = 6;
        custom_hsv_edge_features(target_image, custom_part, H_BINS, S_BINS);

        const int hist_len = H_BINS * S_BINS;
        const int edge_len = 1;
        const int emb_len  = 512;
        const int expected = hist_len + edge_len + emb_len;

        std::string target_base = basename_only(std::string(target_filename));
        int target_idx = -1;
        for (size_t i = 0; i < filenames.size(); i++) {
            if (strcmp(filenames[i], target_filename) == 0 ||
                strcmp(filenames[i], target_base.c_str()) == 0) {
                target_idx = (int)i;
                break;
            }
        }
        if (target_idx < 0) {
            printf("Error: Could not find target in custom CSV\n");
            return -1;
        }
        if ((int)database_features[target_idx].size() != expected) {
            printf("Error: Custom feature vector size mismatch\n");
            return -1;
        }

        target_features.clear();
        target_features.reserve(expected);
        target_features.insert(target_features.end(), custom_part.begin(), custom_part.end());
        target_features.insert(target_features.end(),
                               database_features[target_idx].begin() + (hist_len + edge_len),
                               database_features[target_idx].end());
    } else {
        printf("Error: Unknown method '%s'\n", method.c_str());
        return -1;
    }

    std::vector<ImageDistance> distances;
    distances.reserve(filenames.size());

    for (size_t i = 0; i < filenames.size(); i++) {
        float dist = 0.0f;

        if (method == "baseline") dist = ssd_distance(target_features, database_features[i]);
        else if (method == "histogram") dist = histogram_intersection(target_features, database_features[i]);
        else if (method == "multi_histogram") dist = multi_histogram_distance(target_features, database_features[i]);
        else if (method == "texture_color") dist = texture_color_distance(target_features, database_features[i], bins, 16);
        else if (method == "embedding") dist = cosine_distance(target_features, database_features[i]);
        else if (method == "custom") dist = custom_distance(target_features, database_features[i], 18, 6);
        else if (method == "banana") dist = banana_distance(target_features, database_features[i]);
        else if (method == "bluebin") dist = bluebin_distance(target_features, database_features[i]);
        else if (method == "face") dist = face_metric_distance(target_features, database_features[i]);

        distances.push_back({filenames[i], dist});
    }

    std::sort(distances.begin(), distances.end(), compareDistance);

    int W = N;
    if (W > (int)distances.size()) W = (int)distances.size();

    std::string image_dir = "./data/olympus";

    int tile_w = 220;
    int tile_h = 220;
    int cols = N + 1;
    int rows = 2;

    cv::Mat canvas(rows * tile_h, cols * tile_w, CV_8UC3, cv::Scalar(30,30,30));

    cv::Mat target = resize_or_blank(target_image, tile_w, tile_h);
    draw_label(target, "TARGET");
    target.copyTo(canvas(cv::Rect(0, 0, tile_w, tile_h)));

    for (int i = 0; i < N && i < (int)distances.size(); i++) {
        std::string fname = distances[i].filename;
        cv::Mat img = load_db_image(fname, image_dir);
        img = resize_or_blank(img, tile_w, tile_h);

        char label[128];
        snprintf(label, sizeof(label), "TOP %d (%.3f)", i+1, distances[i].distance);
        draw_label(img, label);

        img.copyTo(canvas(cv::Rect((i+1)*tile_w, 0, tile_w, tile_h)));
    }

    for (int i = 0; i < W; i++) {
        int idx = (int)distances.size() - 1 - i;
        std::string fname = distances[idx].filename;
        cv::Mat img = load_db_image(fname, image_dir);
        img = resize_or_blank(img, tile_w, tile_h);

        char label[128];
        snprintf(label, sizeof(label), "LEAST %d (%.3f)", i+1, distances[idx].distance);
        draw_label(img, label);

        img.copyTo(canvas(cv::Rect((i+1)*tile_w, tile_h, tile_w, tile_h)));
    }

    cv::imshow("CBIR Results", canvas);
    printf("\nPress any key on the image window to close...\n");
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
