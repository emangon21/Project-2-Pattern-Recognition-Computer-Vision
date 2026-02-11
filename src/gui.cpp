#include <opencv2/opencv.hpp>
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
    if (img.empty()) return cv::Mat(h, w, CV_8UC3, cv::Scalar(30,30,30));
    cv::Mat r;
    cv::resize(img, r, cv::Size(w, h));
    return r;
}

static void draw_label(cv::Mat &img, const std::string &text) {
    cv::rectangle(img, cv::Point(0,0), cv::Point(img.cols, 22), cv::Scalar(0,0,0), cv::FILLED);
    cv::putText(img, text, cv::Point(5,16),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,255), 1);
}

struct ImageDistance {
    std::string filename;
    float distance;
};

static bool compareDistance(const ImageDistance &a, const ImageDistance &b) {
    return a.distance < b.distance;
}

static float safe_cosine_distance(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size() || a.empty()) return 1.0f;

    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double x = a[i], y = b[i];
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if (na <= 1e-12 || nb <= 1e-12) return 1.0f;

    double cosv = dot / (std::sqrt(na) * std::sqrt(nb));
    if (cosv > 1.0) cosv = 1.0;
    if (cosv < -1.0) cosv = -1.0;
    return (float)(1.0 - cosv);
}

static float distance_by_method(const char *method,
                                const std::vector<float> &t,
                                const std::vector<float> &d,
                                int bins) {
    if (strcmp(method, "baseline") == 0) {
        return ssd_distance(t, d);
    } else if (strcmp(method, "histogram") == 0) {
        return histogram_intersection(t, d);
    } else if (strcmp(method, "multi_histogram") == 0) {
        return multi_histogram_distance(t, d);
    } else if (strcmp(method, "texture_color") == 0) {
        return texture_color_distance(t, d, bins, 16);
    } else if (strcmp(method, "embedding") == 0) {
        return safe_cosine_distance(t, d);
    } else if (strcmp(method, "face") == 0) {
        return face_metric_distance(t, d);
    }
    return 1.0f;
}

static bool method_needs_csv_target(const char *method) {
    return (strcmp(method, "embedding") == 0 || strcmp(method, "face") == 0);
}

static bool compute_target_features(const std::string &target_path,
                                    const char *method,
                                    int bins,
                                    const std::vector<std::string> &db_names,
                                    const std::vector<std::vector<float>> &db_feats,
                                    std::vector<float> &target_feats) {
    target_feats.clear();

    if (method_needs_csv_target(method)) {
        std::string tb = basename_only(target_path);
        int idx = -1;
        for (int i = 0; i < (int)db_names.size(); i++) {
            if (db_names[i] == target_path || db_names[i] == tb) { idx = i; break; }
        }
        if (idx < 0) return false;
        target_feats = db_feats[idx];
        return true;
    }

    cv::Mat img = cv::imread(target_path);
    if (img.empty()) return false;

    if (strcmp(method, "baseline") == 0) {
        return baseline_features(img, target_feats) == 0;
    } else if (strcmp(method, "histogram") == 0) {
        return histogram_features(img, target_feats, bins) == 0;
    } else if (strcmp(method, "multi_histogram") == 0) {
        return multi_histogram_features(img, target_feats, bins) == 0;
    } else if (strcmp(method, "texture_color") == 0) {
        return texture_color_features(img, target_feats, bins, 16) == 0;
    }

    return false;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <image_dir> <feature_csv>\n", argv[0]);
        printf("Example: %s ./data/olympus ./data/resnet18_embeddings.csv\n", argv[0]);
        return -1;
    }

    std::string img_dir = argv[1];
    std::string csv_file = argv[2];

    std::vector<char*> fn_raw;
    std::vector<std::vector<float>> db_feats_raw;

    if (read_image_data_csv((char*)csv_file.c_str(), fn_raw, db_feats_raw, 0) != 0) {
        printf("Error: Could not read feature database: %s\n", csv_file.c_str());
        return -1;
    }

    std::vector<std::string> db_names;
    db_names.reserve(fn_raw.size());
    for (auto p : fn_raw) db_names.push_back(std::string(p));

    std::vector<std::string> methods = {
        "embedding",
        "texture_color",
        "multi_histogram",
        "histogram",
        "baseline",
        "face"
    };

    int method_idx = 0;
    int topN = 5;
    int bins = 8;
    int showLeast = 1;

    int target_idx = 0;

    const std::string win = "CBIR GUI";
    cv::namedWindow(win, cv::WINDOW_NORMAL);

    cv::createTrackbar("method", win, &method_idx, (int)methods.size() - 1);
    cv::createTrackbar("topN", win, &topN, 15);
    cv::createTrackbar("bins", win, &bins, 32);
    cv::createTrackbar("showLeast", win, &showLeast, 1);

    printf("Keys: n=next, p=prev, r=recompute, q/ESC=quit\n");

    int last_m = -1, last_n = -1, last_b = -1, last_l = -1, last_t = -1;

    while (true) {
        method_idx = std::min(std::max(method_idx, 0), (int)methods.size() - 1);
        topN = std::max(topN, 1);
        bins = std::max(bins, 1);

        const char *method = methods[method_idx].c_str();

        bool need_recompute = (last_m != method_idx || last_n != topN || last_b != bins || last_l != showLeast || last_t != target_idx);

        static cv::Mat canvas;
        static std::vector<ImageDistance> dists;

        if (need_recompute) {
            last_m = method_idx; last_n = topN; last_b = bins; last_l = showLeast; last_t = target_idx;

            std::string target_path = img_dir + "/" + db_names[target_idx];

            std::vector<float> target_feats;
            bool ok = compute_target_features(target_path, method, bins, db_names, db_feats_raw, target_feats);

            if (!ok) {
                canvas = cv::Mat(240, 900, CV_8UC3, cv::Scalar(30,30,30));
                cv::putText(canvas, "Could not compute target features for this method/CSV",
                            cv::Point(15, 120), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,255), 1);
                cv::imshow(win, canvas);
            } else {
                dists.clear();
                dists.reserve(db_names.size());

                for (size_t i = 0; i < db_names.size(); i++) {
                    float dist = distance_by_method(method, target_feats, db_feats_raw[i], bins);
                    dists.push_back({db_names[i], dist});
                }

                std::sort(dists.begin(), dists.end(), compareDistance);

                int K = std::min(topN, (int)dists.size());
                int L = showLeast ? K : 0;

                int tile_w = 220, tile_h = 220;
                int cols = 1 + K;
                int rows = showLeast ? 2 : 1;

                canvas = cv::Mat(rows * tile_h, cols * tile_w, CV_8UC3, cv::Scalar(25,25,25));

                cv::Mat tgt = resize_or_blank(cv::imread(target_path), tile_w, tile_h);
                draw_label(tgt, "TARGET");
                tgt.copyTo(canvas(cv::Rect(0, 0, tile_w, tile_h)));

                for (int i = 0; i < K; i++) {
                    cv::Mat img = resize_or_blank(load_db_image(dists[i].filename, img_dir), tile_w, tile_h);
                    char label[128];
                    snprintf(label, sizeof(label), "TOP %d (%.3f)", i+1, dists[i].distance);
                    draw_label(img, label);
                    img.copyTo(canvas(cv::Rect((i+1)*tile_w, 0, tile_w, tile_h)));
                }

                if (showLeast) {
                    for (int i = 0; i < L; i++) {
                        int idx = (int)dists.size() - 1 - i;
                        cv::Mat img = resize_or_blank(load_db_image(dists[idx].filename, img_dir), tile_w, tile_h);
                        char label[128];
                        snprintf(label, sizeof(label), "LEAST %d (%.3f)", i+1, dists[idx].distance);
                        draw_label(img, label);
                        img.copyTo(canvas(cv::Rect((i+1)*tile_w, tile_h, tile_w, tile_h)));
                    }
                }

                cv::imshow(win, canvas);
                printf("GUI: method=%s | target=%s | topN=%d | bins=%d | showLeast=%d\n",
                       method, basename_only(target_path).c_str(), topN, bins, showLeast);
            }
        }

        int key = cv::waitKey(30);
        if (key == 27 || key == 'q') break;
        if (key == 'n') {
            target_idx = (target_idx + 1) % (int)db_names.size();
        } else if (key == 'p') {
            target_idx--;
            if (target_idx < 0) target_idx = (int)db_names.size() - 1;
        } else if (key == 'r') {
            last_m = -1;
        }
    }

    cv::destroyAllWindows();
    return 0;
}
