/*
  Method B: LBP Texture Histogram builder

  Usage:
    ./bin/lbp <image_directory> <output_csv>

  Example:
    ./bin/lbp data/olympus data/lbp_features.csv
*/

#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstring>
#include <dirent.h>

#include "features.h"
#include "csv_util.h"

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <image_directory> <output_csv>\n", argv[0]);
        return -1;
    }

    char *dirname = argv[1];
    char *csv_filename = argv[2];

    DIR *dirp = opendir(dirname);
    if (!dirp) {
        printf("Error: Cannot open directory %s\n", dirname);
        return -1;
    }

    printf("Reading images from: %s\n", dirname);
    printf("Writing features to: %s\n\n", csv_filename);

    int reset_file = 1;
    int count = 0;

    struct dirent *dp;
    while ((dp = readdir(dirp)) != NULL) {
        if (strstr(dp->d_name, ".jpg") || strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") || strstr(dp->d_name, ".tif")) {

            char path[512];
            strcpy(path, dirname);
            strcat(path, "/");
            strcat(path, dp->d_name);

            cv::Mat img = cv::imread(path);
            if (img.empty()) {
                printf("Warning: could not read %s\n", path);
                continue;
            }

            std::vector<float> feat;
            if (lbp_histogram_features(img, feat) != 0) {
                printf("Warning: LBP failed on %s\n", path);
                continue;
            }

            if (append_image_data_csv(csv_filename, dp->d_name, feat, reset_file) != 0) {
                printf("Error: Could not write CSV\n");
                closedir(dirp);
                return -1;
            }

            reset_file = 0;
            count++;
        }
    }

    closedir(dirp);
    printf("\nProcessed %d images\n", count);
    return 0;
}
