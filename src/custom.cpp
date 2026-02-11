#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>

#include "features.h"
#include "csv_util.h"

static std::string basename_only(const std::string &path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return path;
    return path.substr(pos + 1);
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Usage: %s <image_dir> <embeddings_csv> <output_csv>\n", argv[0]);
        return -1;
    }

    char *image_dir = argv[1];
    char *emb_csv = argv[2];
    char *out_csv = argv[3];

    const int H_BINS = 18;
    const int S_BINS = 6;

    std::vector<char*> emb_names;
    std::vector<std::vector<float>> emb_feats;

    if (read_image_data_csv(emb_csv, emb_names, emb_feats, 0) != 0) {
        printf("Error: could not read embeddings csv\n");
        return -1;
    }

    std::unordered_map<std::string, std::vector<float>> emb_map;
    emb_map.reserve(emb_names.size());

    for (size_t i = 0; i < emb_names.size(); i++) {
        emb_map[std::string(emb_names[i])] = emb_feats[i];
    }

    int reset_file = 1;
    int count = 0;

    for (const auto &entry : std::filesystem::directory_iterator(image_dir)) {
        if (!entry.is_regular_file()) continue;

        std::string path = entry.path().string();
        std::string base = basename_only(path);

        if (entry.path().extension() != ".jpg" && entry.path().extension() != ".JPG") continue;

        auto it = emb_map.find(base);
        if (it == emb_map.end()) continue;

        cv::Mat img = cv::imread(path);
        if (img.empty()) continue;

        std::vector<float> custom_part;
        if (custom_hsv_edge_features(img, custom_part, H_BINS, S_BINS) != 0) continue;

        std::vector<float> full;
        full.reserve(custom_part.size() + it->second.size());
        full.insert(full.end(), custom_part.begin(), custom_part.end());
        full.insert(full.end(), it->second.begin(), it->second.end());

        char *fname_c = (char*)base.c_str();
        append_image_data_csv(out_csv, fname_c, full, reset_file);
        reset_file = 0;
        count++;
    }

    printf("Wrote custom features for %d images to %s\n", count, out_csv);
    return 0;
}
