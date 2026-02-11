#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

#include "features.h"
#include "csv_util.h"

static std::string basename_only(const std::string &path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return path;
    return path.substr(pos + 1);
}

static bool is_image_file(const std::string &p) {
    std::string s = p;
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return (s.find(".jpg") != std::string::npos ||
            s.find(".jpeg") != std::string::npos ||
            s.find(".png") != std::string::npos ||
            s.find(".bmp") != std::string::npos ||
            s.find(".tif") != std::string::npos ||
            s.find(".tiff") != std::string::npos);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s <image_directory> <output_csv>\n", argv[0]);
        return -1;
    }

    std::string img_dir = argv[1];
    std::string out_csv = argv[2];

    // Collect image paths
    std::vector<cv::String> files;
    cv::glob(img_dir + "/*", files, false);

    if (files.empty()) {
        printf("Error: No files found in %s\n", img_dir.c_str());
        return -1;
    }

    // Build CSV rows
    std::vector<std::vector<std::string>> rows;
    rows.reserve(files.size());

    int kept = 0;
    for (const auto &f : files) {
        std::string path = std::string(f);
        if (!is_image_file(path)) continue;

        cv::Mat img = cv::imread(path);
        if (img.empty()) continue;

        std::vector<float> feat;
        banana_feature_vector(img, feat);

        // Write basename into CSV so query display works reliably
        std::vector<std::string> row;
        row.push_back(basename_only(path));
        row.reserve(1 + feat.size());

        char buf[64];
        for (float v : feat) {
            std::snprintf(buf, sizeof(buf), "%.6f", v);
            row.push_back(buf);
        }

        rows.push_back(row);
        kept++;
    }

    if (kept == 0) {
        printf("Error: No readable images in %s\n", img_dir.c_str());
        return -1;
    }

    // Write CSV
    int rc = write_csv(out_csv.c_str(), rows);
    if (rc != 0) {
        printf("Error: Failed to write CSV %s\n", out_csv.c_str());
        return -1;
    }

    printf("Wrote %d rows to %s\n", kept, out_csv.c_str());
    return 0;
}
