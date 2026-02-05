/*
  Query program - Find top N similar images
  
  Usage: ./bin/query <target_image> <feature_csv> <method> <N>
*/
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include "features.h"
#include "csv_util.h"

//structure to hold image filename and its distance
struct ImageDistance {
    char *filename;
    float distance;
};

//comparing for sorting by distance
bool compareDistance(const ImageDistance &a, const ImageDistance &b) {
    return a.distance < b.distance;
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        printf("Usage: %s <target_image> <feature_csv> <method> <N>\n", argv[0]);
        printf("  method: baseline (for Task 1)\n");
        printf("  N: number of top matches to return\n");
        printf("Example: %s ./data/olympus/pic.1016.jpg features_baseline.csv baseline 4\n", argv[0]);
        return -1;
    }
    
    char *target_filename = argv[1];
    char *csv_filename = argv[2];
    char *method = argv[3];
    int N = atoi(argv[4]);
    
    printf("Target image: %s\n", target_filename);
    printf("Feature file: %s\n", csv_filename);
    printf("Method: %s\n", method);
    printf("Top N matches: %d\n\n", N);
    
    //read target image
    cv::Mat target_image = cv::imread(target_filename);
    if (target_image.empty()) {
        printf("Error: Could not read target image %s\n", target_filename);
        return -1;
    }
    
    //extract features from target image
    std::vector<float> target_features;
    if (strcmp(method, "baseline") == 0) {
        if (baseline_features(target_image, target_features) != 0) {
            printf("Error: Could not extract features from target image\n");
            return -1;
        }
        printf("Extracted %lu features from target image\n", target_features.size());
    } else {
        printf("Error: Unknown method '%s'\n", method);
        return -1;
    }
    
    //read feature database
    std::vector<char *> filenames;
    std::vector<std::vector<float>> database_features;
    
    if (read_image_data_csv(csv_filename, filenames, database_features, 0) != 0) {
        printf("Error: Could not read feature database\n");
        return -1;
    }
    
    printf("Loaded %lu images from database\n\n", filenames.size());
    
    //compute distances to all images in db
    std::vector<ImageDistance> distances;
    
    for (size_t i = 0; i < filenames.size(); i++) {
        float dist = ssd_distance(target_features, database_features[i]);
        
        ImageDistance img_dist;
        img_dist.filename = filenames[i];
        img_dist.distance = dist;
        distances.push_back(img_dist);
    }
    
    //sort by distance ascendingly
    std::sort(distances.begin(), distances.end(), compareDistance);
    
    //print top N matches
    printf("Top %d matches:\n", N);
    printf("%-30s %s\n", "Filename", "Distance");
    printf("%-30s %s\n", "--------", "--------");
    
    for (int i = 0; i < N && i < distances.size(); i++) {
        printf("%-30s %.2f\n", distances[i].filename, distances[i].distance);
    }
    
    return 0;
}