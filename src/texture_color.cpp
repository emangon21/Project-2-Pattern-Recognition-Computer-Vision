/*
  Texture and color matching - Task 4
  Extracts RGB color histogram + Sobel magnitude texture histogram
  
  Usage: ./bin/texture_color <image_directory> <output_csv> [color_bins] [texture_bins]
*/
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include "features.h"
#include "csv_util.h"

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <image_directory> <output_csv> [color_bins] [texture_bins]\n", argv[0]);
        printf("  color_bins: bins per RGB channel (default: 8)\n");
        printf("  texture_bins: bins for gradient magnitude (default: 16)\n");
        printf("Example: %s ./data/olympus features_texture_color.csv 8 16\n", argv[0]);
        return -1;
    }
    
    char *dirname = argv[1];
    char *csv_filename = argv[2];
    int color_bins = 8;
    int texture_bins = 16;
    
    if (argc >= 4) {
        color_bins = atoi(argv[3]);
    }
    if (argc >= 5) {
        texture_bins = atoi(argv[4]);
    }
    
    printf("Processing directory: %s\n", dirname);
    printf("Output CSV: %s\n", csv_filename);
    printf("Color bins per channel: %d (total: %d)\n", color_bins, color_bins * color_bins * color_bins);
    printf("Texture bins: %d\n", texture_bins);
    printf("Total features per image: %d\n\n", color_bins * color_bins * color_bins + texture_bins);
    
    DIR *dirp = opendir(dirname);
    if (!dirp) {
        printf("Error: Cannot open directory %s\n", dirname);
        return -1;
    }
    
    int reset_file = 1;
    int image_count = 0;
    struct dirent *dp;
    
    //loop over all files in directory
    while ((dp = readdir(dirp)) != NULL) {
        //check if file is an image
        if (strstr(dp->d_name, ".jpg") || strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") || strstr(dp->d_name, ".tif")) {
            
            //build full path
            char buffer[512];
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);
            
            printf("Processing: %s\n", dp->d_name);
            
            //read image
            cv::Mat image = cv::imread(buffer);
            if (image.empty()) {
                printf("Warning: Could not read %s\n", buffer);
                continue;
            }
            
            //extract texture + color features
            std::vector<float> features;
            if (texture_color_features(image, features, color_bins, texture_bins) != 0) {
                printf("Warning: Could not extract features from %s\n", buffer);
                continue;
            }
            
            //write to CSV
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