/*
  Baseline matching - Task 1
  Extracts 7x7 center square features and saves to CSV
  
  Usage: ./bin/baseline <image_directory> <output_csv>
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
        printf("Example: %s ./data/olympus features_baseline.csv\n", argv[0]);
        return -1;
    }
    
    char *dirname = argv[1];
    char *csv_filename = argv[2];
    
    printf("Processing directory: %s\n", dirname);
    printf("Output CSV: %s\n", csv_filename);
    
    DIR *dirp = opendir(dirname);
    if (!dirp) {
        printf("Error: Cannot open directory %s\n", dirname);
        return -1;
    }
    
    int reset_file = 1; //reset csv file on first write
    int image_count = 0;
    struct dirent *dp;
    
    //loop all files in directory
    while ((dp = readdir(dirp)) != NULL) {
        //check if file is image
        if (strstr(dp->d_name, ".jpg") || strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") || strstr(dp->d_name, ".tif")) {
            
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
            
            //extract baseline features
            std::vector<float> features;
            if (baseline_features(image, features) != 0) {
                printf("Warning: Could not extract features from %s\n", buffer);
                continue;
            }
            
            //write to csv
            if (append_image_data_csv(csv_filename, dp->d_name, features, reset_file) != 0) {
                printf("Error: Could not write to CSV\n");
                closedir(dirp);
                return -1;
            }
            
            reset_file = 0; //only reset on first write
            image_count++;
        }
    }
    
    closedir(dirp);
    printf("\nProcessed %d images\n", image_count);
    printf("Features saved to %s\n", csv_filename);
    
    return 0;
}