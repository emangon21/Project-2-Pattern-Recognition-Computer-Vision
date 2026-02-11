#include <opencv2/opencv.hpp>
#include <cstdio>
#include <vector>
#include <string>
#include <filesystem>
#include <unordered_map>
#include <algorithm>

#include "features.h"
#include "csv_util.h"

namespace fs = std::filesystem;

static std::string basename_only(const std::string &path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return path;
    return path.substr(pos + 1);
}

static bool try_load_face_cascade(cv::CascadeClassifier &cc) {
    // Common OpenCV locations on macOS/Homebrew + fallback to local data folder
    const char* candidates[] = {
        "./data/haarcascade_frontalface_default.xml",
        "/opt/homebrew/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/opt/homebrew/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/local/share/opencv/haarcascades/haarcascade_frontalface_default.xml"
    };

    for (auto p : candidates) {
        if (cc.load(p)) {
            printf("Loaded face cascade: %s\n", p);
            return true;
        }
    }
    return false;
}

// returns true if face found; face_rect filled with largest face
static bool detect_largest_face(const cv::Mat &img, cv::CascadeClassifier &cc, cv::Rect &face_rect) {
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);

    std::vector<cv::Rect> faces;
    cc.detectMultiScale(gray, faces, 1.1, 4, 0, cv::Size(40, 40));

    if (faces.empty()) return false;

    int best = 0;
    int bestArea = faces[0].area();
    for (int i = 1; i < (int)faces.size(); i++) {
        int a = faces[i].area();
        if (a > bestArea) { bestArea = a; best = i; }
    }
    face_rect = faces[best];
    return true;
}

// 8x8x8 RGB histogram over ROI (512 dims), normalized
static void roi_rgb_hist512(const cv::Mat &roi_bgr, std::vector<float> &hist) {
    const int bins = 8;
    hist.assign(bins * bins * bins, 0.0f);

    int bin_size = 256 / bins;
    int total = roi_bgr.rows * roi_bgr.cols;
    if (total <= 0) return;

    for (int r = 0; r < roi_bgr.rows; r++) {
        const cv::Vec3b *p = roi_bgr.ptr<cv::Vec3b>(r);
        for (int c = 0; c < roi_bgr.cols; c++) {
            int b = p[c][0] / bin_size; if (b >= bins) b = bins - 1;
            int g = p[c][1] / bin_size; if (g >= bins) g = bins - 1;
            int rr = p[c][2] / bin_size; if (rr >= bins) rr = bins - 1;

            int idx = rr * bins * bins + g * bins + b;
            hist[idx] += 1.0f;
        }
    }

    for (float &v : hist) v /= (float)total;
}

static std::unordered_map<std::string, std::vector<float>>
load_embeddings_map(const std::string &emb_csv) {
    std::unordered_map<std::string, std::vector<float>> mp;

    std::vector<char*> fn;
    std::vector<std::vector<float>> data;

    if (read_image_data_csv((char*)emb_csv.c_str(), fn, data, 0) != 0) {
        printf("Error: could not read embeddings CSV: %s\n", emb_csv.c_str());
        return mp;
    }

    for (size_t i = 0; i < fn.size(); i++) {
        std::string name(fn[i]);
        mp[name] = data[i];
        delete [] fn[i];
    }
    return mp;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Usage: %s <image_dir> <resnet18_embeddings_csv> <output_csv>\n", argv[0]);
        return -1;
    }

    std::string image_dir = argv[1];
    std::string emb_csv   = argv[2];
    std::string out_csv   = argv[3];

    cv::CascadeClassifier face_cc;
    if (!try_load_face_cascade(face_cc)) {
        printf("ERROR: Could not load Haar face cascade.\n");
        printf("Fix: copy haarcascade_frontalface_default.xml into ./data/ and rerun.\n");
        return -1;
    }

    auto emb_map = load_embeddings_map(emb_csv);
    if (emb_map.empty()) {
        printf("WARNING: embedding map empty; face features will still be written but embeddings will be zeros.\n");
    }

    int reset_file = 1;
    int count = 0;

    for (const auto &entry : fs::directory_iterator(image_dir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".jpg") continue;

        std::string base = entry.path().filename().string();
        cv::Mat img = cv::imread(entry.path().string());
        if (img.empty()) continue;

        // --- face stats + face hist ---
        float has_face = 0.0f;
        float area_ratio = 0.0f, cx = 0.0f, cy = 0.0f, aspect = 0.0f;

        std::vector<float> face_hist;
        face_hist.assign(512, 0.0f);

        cv::Rect face;
        if (detect_largest_face(img, face_cc, face)) {
            has_face = 1.0f;

            area_ratio = (float)face.area() / (float)(img.rows * img.cols);
            cx = (face.x + face.width * 0.5f) / (float)img.cols;
            cy = (face.y + face.height * 0.5f) / (float)img.rows;
            aspect = (face.height > 0) ? ((float)face.width / (float)face.height) : 0.0f;

            cv::Rect clipped = face & cv::Rect(0, 0, img.cols, img.rows);
            cv::Mat roi = img(clipped).clone();
            roi_rgb_hist512(roi, face_hist);
        }

        // --- embedding (512) ---
        std::vector<float> emb(512, 0.0f);
        auto it = emb_map.find(base);
        if (it != emb_map.end() && it->second.size() == 512) {
            emb = it->second;
        }

        // final vector: 5 + 512 + 512 = 1029
        std::vector<float> feat;
        feat.reserve(1029);
        feat.push_back(has_face);
        feat.push_back(area_ratio);
        feat.push_back(cx);
        feat.push_back(cy);
        feat.push_back(aspect);
        feat.insert(feat.end(), face_hist.begin(), face_hist.end());
        feat.insert(feat.end(), emb.begin(), emb.end());

        append_image_data_csv((char*)out_csv.c_str(), (char*)base.c_str(), feat, reset_file);
        reset_file = 0;
        count++;
    }

    printf("Wrote face features for %d images -> %s\n", count, out_csv.c_str());
    return 0;
}
