/*
  Histogram matching - Task 2
  Extracts RGB color histogram features and saves to CSV
  
  Usage: ./bin/histogram <image_directory> <output_csv> [bins]
*/
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include "features.h"
#include "csv_util.h"

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <image_directory> <output_csv> [bins]\n", argv[0]);
        printf("  bins: number of bins per channel (default: 8)\n");
        printf("Example: %s ./data/olympus features_histogram.csv 8\n", argv[0]);
        return -1;
    }
    
    char *dirname = argv[1];
    char *csv_filename = argv[2];
    int bins = 8; 
    
    if (argc >= 4) {
        bins = atoi(argv[3]);
    }
    
    printf("Processing directory: %s\n", dirname);
    printf("Output CSV: %s\n", csv_filename);
    printf("Bins per channel: %d\n", bins);
    printf("Total histogram bins: %d\n\n", bins * bins * bins);
    
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
        //check if file is image
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
            
            //extract histogram features
            std::vector<float> features;
            if (histogram_features(image, features, bins) != 0) {
                printf("Warning: Could not extract features from %s\n", buffer);
                continue;
            }
            
            //write to csv
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