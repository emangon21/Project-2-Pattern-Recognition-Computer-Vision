/*
  HSV Spatial Moments (Extension Method A)

  Splits the image into a grid (default 3x3). For each cell, computes
  mean and standard deviation for H, S, and V (6 values per cell).

  Total feature dims = grid*grid*6 (for 3x3 => 54).

  Usage:
    ./bin/hsv_moments <image_directory> <output_csv> [grid]

  Example:
    ./bin/hsv_moments ./data/olympus ./data/features_hsv_moments.csv 3
*/

#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstring>
#include <dirent.h>

#include "features.h"
#include "csv_util.h"

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <image_directory> <output_csv> [grid]\n", argv[0]);
        printf("  grid: number of cells per dimension (default: 3 => 3x3)\n");
        printf("Example: %s ./data/olympus ./data/features_hsv_moments.csv 3\n", argv[0]);
        return -1;
    }

    char *dirname = argv[1];
    char *csv_filename = argv[2];
    int grid = (argc >= 4) ? atoi(argv[3]) : 3;
    if (grid <= 0) grid = 3;

    printf("Processing directory: %s\n", dirname);
    printf("Output CSV: %s\n", csv_filename);
    printf("Grid: %dx%d (feature dims = %d)\n\n", grid, grid, grid * grid * 6);

    DIR *dirp = opendir(dirname);
    if (!dirp) {
        printf("Error: Cannot open directory %s\n", dirname);
        return -1;
    }

    int reset_file = 1;
    int image_count = 0;
    struct dirent *dp;

    while ((dp = readdir(dirp)) != NULL) {
        if (strstr(dp->d_name, ".jpg") || strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") || strstr(dp->d_name, ".tif")) {

            char buffer[512];
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);

            cv::Mat image = cv::imread(buffer);
            if (image.empty()) {
                printf("Warning: Could not read %s\n", buffer);
                continue;
            }

            std::vector<float> features;
            if (hsv_spatial_moments(image, features, grid) != 0) {
                printf("Warning: Could not extract hsv moments from %s\n", buffer);
                continue;
            }

            if (append_image_data_csv(csv_filename, dp->d_name, features, reset_file) != 0) {
                printf("Error: Could not write to CSV\n");
                closedir(dirp);
                return -1;
            }

            reset_file = 0;
            image_count++;
        }
    }

    closedir(dirp);
    printf("\nProcessed %d images\n", image_count);
    printf("Features saved to %s\n", csv_filename);
    return 0;
}
