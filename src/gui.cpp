#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <filesystem>

#include "features.h"
#include "csv_util.h"

namespace fs = std::filesystem;

static std::string basename_only(const std::string &path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return path;
    return path.substr(pos + 1);
}

static bool file_exists(const std::string &p) {
    return fs::exists(p) && fs::is_regular_file(p);
}

static std::vector<std::string> list_jpg_basenames(const std::string &dir) {
    std::vector<std::string> out;
    for (const auto &entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() == ".jpg") {
            out.push_back(entry.path().filename().string());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

static cv::Mat safe_imread(const std::string &fullpath) {
    cv::Mat img = cv::imread(fullpath);
    return img;
}

static cv::Mat make_thumbnail(const cv::Mat &img, int w, int h) {
    if (img.empty()) return cv::Mat(h, w, CV_8UC3, cv::Scalar(0,0,0));
    cv::Mat out;
    cv::resize(img, out, cv::Size(w, h));
    return out;
}

struct DB {
    std::vector<std::string> names;               // basenames
    std::vector<std::vector<float>> feats;
    std::unordered_map<std::string, int> idx;
    bool loaded = false;
};

static void build_index(DB &db) {
    db.idx.clear();
    for (int i = 0; i < (int)db.names.size(); i++) {
        db.idx[db.names[i]] = i;
    }
}

static bool load_csv_db(const std::string &csv_path, DB &db) {
    db = DB();
    if (!file_exists(csv_path)) return false;

    std::vector<char*> fn;
    std::vector<std::vector<float>> data;

    char *cpath = (char*)csv_path.c_str();
    if (read_image_data_csv(cpath, fn, data, 0) != 0) {
        return false;
    }

    db.names.reserve(fn.size());
    for (char* p : fn) {
        db.names.push_back(std::string(p));
        delete [] p; // important: csv_util allocates these
    }
    db.feats = std::move(data);
    build_index(db);
    db.loaded = true;
    return true;
}

// ---------- baseline DB: compute once from images (since no baseline CSV is in your repo) ----------
static bool build_baseline_db_from_images(const std::string &img_dir,
                                         const std::vector<std::string> &all_imgs,
                                         DB &db) {
    db = DB();
    db.names = all_imgs;
    db.feats.resize(all_imgs.size());
    for (int i = 0; i < (int)all_imgs.size(); i++) {
        cv::Mat img = safe_imread(img_dir + "/" + all_imgs[i]);
        if (img.empty()) {
            db.feats[i] = std::vector<float>(147, 0.0f);
            continue;
        }
        std::vector<float> f;
        // baseline_features expects non-const Mat&
        cv::Mat tmp = img;
        if (baseline_features(tmp, f) != 0) {
            db.feats[i] = std::vector<float>(147, 0.0f);
        } else {
            db.feats[i] = std::move(f);
        }
    }
    build_index(db);
    db.loaded = true;
    return true;
}

// ---------- banana features (same as your query.cpp) ----------
static float banana_edge_density_local(const cv::Mat &src) {
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

    int best = -1, bestArea = 0;
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
                var_x += dx*dx;
                var_y += dy*dy;
            }
            var_x /= pts.size();
            var_y /= pts.size();
            var_x /= (double)(img.cols * img.cols);
            var_y /= (double)(img.rows * img.rows);
        }
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

    float ed = banana_edge_density_local(img);

    feat = { area_ratio, cx_n, cy_n, (float)var_x, (float)var_y, ed, bbox_aspect, extent, solidity };
    return 0;
}

// ---------- bluebin features (same as your updated query.cpp) ----------
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

    int best = -1, bestArea = 0;
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
                var_x += dx*dx;
                var_y += dy*dy;
            }
            var_x /= pts.size();
            var_y /= pts.size();
            var_x /= (double)(img.cols * img.cols);
            var_y /= (double)(img.rows * img.rows);
        }
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

    feat = { area_ratio, cx_n, cy_n, (float)var_x, (float)var_y, ed, bbox_aspect, extent, solidity };
    return 0;
}

// ---------- GUI state ----------
static std::string g_img_dir;
static std::vector<std::string> g_all_imgs;
static int g_qidx = 0;

static int g_method = 0; // trackbar index
static int g_topN = 5;
static int g_showWorst = 0;

static const char* METHOD_NAMES[] = {
    "baseline",
    "histogram",
    "multi_histogram",
    "texture_color",
    "embedding",
    "custom",
    "banana",
    "bluebin"
};
static const int METHOD_COUNT = 8;

static DB db_baseline;
static DB db_hist;
static DB db_multi;
static DB db_texcol;
static DB db_embed;
static DB db_custom;
static DB db_banana;
static DB db_bluebin;

static std::string default_csv_for_method(const std::string &method) {
    if (method == "histogram")       return "features_histogram.csv";
    if (method == "multi_histogram") return "features_multi_histogram.csv";
    if (method == "texture_color")   return "features_texture_color.csv";
    if (method == "embedding")       return "data/resnet18_embeddings.csv";
    if (method == "custom")          return "features_custom.csv";
    if (method == "banana")          return "features_banana.csv";
    if (method == "bluebin")         return "features_bluebin.csv";
    return ""; // baseline computed directly
}

static DB* db_for_method(const std::string &m) {
    if (m == "baseline") return &db_baseline;
    if (m == "histogram") return &db_hist;
    if (m == "multi_histogram") return &db_multi;
    if (m == "texture_color") return &db_texcol;
    if (m == "embedding") return &db_embed;
    if (m == "custom") return &db_custom;
    if (m == "banana") return &db_banana;
    if (m == "bluebin") return &db_bluebin;
    return nullptr;
}

static bool ensure_db_loaded(const std::string &method) {
    DB *db = db_for_method(method);
    if (!db) return false;
    if (db->loaded) return true;

    if (method == "baseline") {
        return build_baseline_db_from_images(g_img_dir, g_all_imgs, *db);
    }

    std::string csv = default_csv_for_method(method);
    if (!file_exists(csv)) {
        // try relative to project root: if user runs from project root this is fine.
        // If not, tell them clearly in console.
        printf("Missing CSV for %s: expected '%s'\n", method.c_str(), csv.c_str());
        return false;
    }
    return load_csv_db(csv, *db);
}

// Map query basename -> db row index (many CSVs store basenames)
static int find_row_index(DB &db, const std::string &query_basename) {
    auto it = db.idx.find(query_basename);
    if (it == db.idx.end()) return -1;
    return it->second;
}

// compute target features (some methods compute from image; embedding/custom load from CSV)
static bool compute_target_features(const std::string &method,
                                   const std::string &query_basename,
                                   std::vector<float> &target) {
    target.clear();

    if (!ensure_db_loaded(method)) return false;
    DB *db = db_for_method(method);

    std::string fullpath = g_img_dir + "/" + query_basename;
    cv::Mat img = safe_imread(fullpath);

    if (method == "baseline") {
        if (img.empty()) return false;
        cv::Mat tmp = img;
        return baseline_features(tmp, target) == 0;
    }

    if (method == "histogram") {
        if (img.empty()) return false;
        cv::Mat tmp = img;
        return histogram_features(tmp, target, 8) == 0; // CSV is 8 bins
    }

    if (method == "multi_histogram") {
        if (img.empty()) return false;
        cv::Mat tmp = img;
        return multi_histogram_features(tmp, target, 8) == 0; // CSV is 8 bins
    }

    if (method == "texture_color") {
        if (img.empty()) return false;
        cv::Mat tmp = img;
        return texture_color_features(tmp, target, 8, 16) == 0; // CSV is 8/16
    }

    if (method == "banana") {
        if (img.empty()) return false;
        return banana_features_local(img, target) == 0;
    }

    if (method == "bluebin") {
        if (img.empty()) return false;
        return bluebin_features_local(img, target) == 0;
    }

    if (method == "embedding") {
        int idx = find_row_index(*db, query_basename);
        if (idx < 0) return false;
        target = db->feats[idx];
        return true;
    }

    if (method == "custom") {
        // custom_distance expects: [HSV+edge] + [512 embedding]
        // Your CSV already stores that full vector; we compute HSV+edge from image and
        // stitch embedding tail from CSV row.
        const int H_BINS = 18, S_BINS = 6;
        const int hist_len = H_BINS * S_BINS;
        const int edge_len = 1;
        const int emb_len = 512;
        const int expected = hist_len + edge_len + emb_len;

        if (img.empty()) return false;

        std::vector<float> custom_part;
        if (custom_hsv_edge_features(img, custom_part, H_BINS, S_BINS) != 0) return false;
        if ((int)custom_part.size() != hist_len + edge_len) return false;

        int idx = find_row_index(*db, query_basename);
        if (idx < 0) return false;
        if ((int)db->feats[idx].size() != expected) return false;

        target.clear();
        target.reserve(expected);
        target.insert(target.end(), custom_part.begin(), custom_part.end());
        target.insert(target.end(),
                      db->feats[idx].begin() + (hist_len + edge_len),
                      db->feats[idx].end());
        return true;
    }

    return false;
}

static float distance_dispatch(const std::string &method,
                               const std::vector<float> &t,
                               const std::vector<float> &d) {
    if (method == "baseline") return ssd_distance(t, d);
    if (method == "histogram") return histogram_intersection(t, d);
    if (method == "multi_histogram") return multi_histogram_distance(t, d);
    if (method == "texture_color") return texture_color_distance(t, d, 8, 16);
    if (method == "embedding") return cosine_distance(t, d);
    if (method == "custom") return custom_distance(t, d, 18, 6);
    if (method == "banana") return banana_distance(t, d);
    if (method == "bluebin") return bluebin_distance(t, d);
    return 1e9f;
}

struct Result {
    std::string name;
    float dist;
};

static std::vector<Result> run_retrieval(const std::string &method,
                                        const std::string &query_basename) {
    std::vector<Result> out;

    if (!ensure_db_loaded(method)) return out;
    DB *db = db_for_method(method);

    std::vector<float> target;
    if (!compute_target_features(method, query_basename, target)) return out;

    out.reserve(db->names.size());
    for (int i = 0; i < (int)db->names.size(); i++) {
        float dist = distance_dispatch(method, target, db->feats[i]);
        out.push_back({ db->names[i], dist });
    }

    std::sort(out.begin(), out.end(), [](const Result &a, const Result &b){
        return a.dist < b.dist;
    });
    return out;
}

// ---------- UI drawing ----------
static cv::Mat draw_results_grid(const std::vector<Result> &results,
                                int start_idx,
                                int count,
                                int thumb_w,
                                int thumb_h,
                                int cols) {
    int rows = (count + cols - 1) / cols;
    cv::Mat canvas(rows * thumb_h, cols * thumb_w, CV_8UC3, cv::Scalar(25,25,25));

    for (int k = 0; k < count; k++) {
        int idx = start_idx + k;
        if (idx < 0 || idx >= (int)results.size()) break;

        int r = k / cols;
        int c = k % cols;

        std::string full = g_img_dir + "/" + results[idx].name;
        cv::Mat img = safe_imread(full);
        cv::Mat t = make_thumbnail(img, thumb_w, thumb_h);

        t.copyTo(canvas(cv::Rect(c * thumb_w, r * thumb_h, thumb_w, thumb_h)));

        // label
        std::string label = results[idx].name;
        cv::putText(canvas, label,
                    cv::Point(c * thumb_w + 6, r * thumb_h + 18),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255,255,255), 1);

        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.3f", results[idx].dist);
        cv::putText(canvas, buf,
                    cv::Point(c * thumb_w + 6, r * thumb_h + thumb_h - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(255,255,255), 1);
    }

    return canvas;
}

struct ClickContext {
    std::vector<Result> results;
    int start_idx = 0;
    int count = 0;
    int thumb_w = 160;
    int thumb_h = 120;
    int cols = 5;
};

static ClickContext g_click_ctx;

static void on_mouse(int event, int x, int y, int, void*) {
    if (event != cv::EVENT_LBUTTONDOWN) return;

    int c = x / g_click_ctx.thumb_w;
    int r = y / g_click_ctx.thumb_h;
    int k = r * g_click_ctx.cols + c;
    if (k < 0 || k >= g_click_ctx.count) return;

    int idx = g_click_ctx.start_idx + k;
    if (idx < 0 || idx >= (int)g_click_ctx.results.size()) return;

    std::string name = g_click_ctx.results[idx].name;
    cv::Mat img = safe_imread(g_img_dir + "/" + name);
    if (img.empty()) return;

    cv::imshow("Selected", img);
}

static void show_help() {
    printf("\n--- GUI Controls ---\n");
    printf("n : next query image\n");
    printf("p : previous query image\n");
    printf("r : run retrieval\n");
    printf("w : toggle show WORST (least similar) instead of top\n");
    printf("ESC/q : quit\n\n");
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <image_dir>\n", argv[0]);
        printf("Run from project root so CSV paths resolve (features_*.csv, data/resnet18_embeddings.csv)\n");
        return -1;
    }

    g_img_dir = argv[1];
    if (!fs::exists(g_img_dir)) {
        printf("Error: image_dir not found: %s\n", g_img_dir.c_str());
        return -1;
    }

    g_all_imgs = list_jpg_basenames(g_img_dir);
    if (g_all_imgs.empty()) {
        printf("No .jpg images found in %s\n", g_img_dir.c_str());
        return -1;
    }

    show_help();

    cv::namedWindow("Query", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Results", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Selected", cv::WINDOW_AUTOSIZE);

    cv::createTrackbar("method (0..7)", "Query", &g_method, METHOD_COUNT - 1);
    cv::createTrackbar("TopN", "Query", &g_topN, 20);
    if (g_topN < 1) g_topN = 5;

    cv::setMouseCallback("Results", on_mouse);

    std::vector<Result> results;
    bool need_run = true;

    while (true) {
        g_method = std::max(0, std::min(g_method, METHOD_COUNT - 1));
        std::string method = METHOD_NAMES[g_method];

        if (g_topN < 1) g_topN = 1;
        int N = g_topN;

        std::string qname = g_all_imgs[g_qidx];
        cv::Mat qimg = safe_imread(g_img_dir + "/" + qname);

        cv::Mat qdisp = qimg.empty() ? cv::Mat(480, 640, CV_8UC3, cv::Scalar(0,0,0)) : qimg.clone();
        cv::putText(qdisp, "Query: " + qname, cv::Point(15, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2);
        cv::putText(qdisp, "Method: " + method, cv::Point(15, 55),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,255,255), 2);
        cv::putText(qdisp, "TopN: " + std::to_string(N) + (g_showWorst ? " (WORST)" : " (TOP)"),
                    cv::Point(15, 85), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(255,255,255), 2);

        cv::imshow("Query", qdisp);

        if (need_run) {
            if (!ensure_db_loaded(method)) {
                // show blank results with message
                cv::Mat blank(240, 800, CV_8UC3, cv::Scalar(25,25,25));
                cv::putText(blank, "Missing/failed to load CSV for method: " + method,
                            cv::Point(20, 120), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                            cv::Scalar(255,255,255), 2);
                cv::imshow("Results", blank);
                results.clear();
            } else {
                results = run_retrieval(method, qname);

                int start_idx = 0;
                if (g_showWorst && !results.empty()) {
                    start_idx = std::max(0, (int)results.size() - N);
                }

                int thumb_w = 160, thumb_h = 120, cols = 5;
                cv::Mat grid = draw_results_grid(results, start_idx, N, thumb_w, thumb_h, cols);
                cv::imshow("Results", grid);

                g_click_ctx.results = results;
                g_click_ctx.start_idx = start_idx;
                g_click_ctx.count = N;
                g_click_ctx.thumb_w = thumb_w;
                g_click_ctx.thumb_h = thumb_h;
                g_click_ctx.cols = cols;
            }
            need_run = false;
        }

        int key = cv::waitKey(30);
        if (key == 27 || key == 'q') break;           // ESC / q
        if (key == 'n') { g_qidx = (g_qidx + 1) % (int)g_all_imgs.size(); need_run = true; }
        if (key == 'p') { g_qidx = (g_qidx - 1 + (int)g_all_imgs.size()) % (int)g_all_imgs.size(); need_run = true; }
        if (key == 'r') { need_run = true; }
        if (key == 'w') { g_showWorst = 1 - g_showWorst; need_run = true; }

        // if user changes trackbars, rerun
        static int last_method = -1, last_topN = -1;
        if (last_method != g_method || last_topN != g_topN) {
            last_method = g_method;
            last_topN = g_topN;
            need_run = true;
        }
    }

    return 0;
}
