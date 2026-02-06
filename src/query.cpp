/*
  Query program - Find top N similar images
  
  Usage: ./bin/query <target_image> <feature_csv> <method> <N> [bins]
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
        printf("Usage: %s <target_image> <feature_csv> <method> <N> [bins]\n", argv[0]);
        printf("  method: baseline, histogram, multi_histogram, texture_color\n");
        printf("  N: number of top matches to return\n");
        printf("  bins: for histogram methods (default: 8)\n");
        printf("Example: %s ./data/olympus/pic.0535.jpg features_texture_color.csv texture_color 4 8\n", argv[0]);
        return -1;
    }
    
    char *target_filename = argv[1];
    char *csv_filename = argv[2];
    char *method = argv[3];
    int N = atoi(argv[4]);
    int bins = 8;
    
    if (argc >= 6) {
        bins = atoi(argv[5]);
    }
    
    printf("Target image: %s\n", target_filename);
    printf("Feature file: %s\n", csv_filename);
    printf("Method: %s\n", method);
    printf("Top N matches: %d\n", N);
    if (strcmp(method, "histogram") == 0 || strcmp(method, "multi_histogram") == 0 || strcmp(method, "texture_color") == 0) {
        printf("Bins per channel: %d\n", bins);
    }
    printf("\n");
    
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
    } else if (strcmp(method, "histogram") == 0) {
        if (histogram_features(target_image, target_features, bins) != 0) {
            printf("Error: Could not extract features from target image\n");
            return -1;
        }
    } else if (strcmp(method, "multi_histogram") == 0) {
        if (multi_histogram_features(target_image, target_features, bins) != 0) {
            printf("Error: Could not extract features from target image\n");
            return -1;
        }
    } else if (strcmp(method, "texture_color") == 0) {
        if (texture_color_features(target_image, target_features, bins, 16) != 0) {
            printf("Error: Could not extract features from target image\n");
            return -1;
        }
    } else {
        printf("Error: Unknown method '%s'\n", method);
        return -1;
    }
    
    printf("Extracted %lu features from target image\n", target_features.size());
    
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
        float dist;
        
        if (strcmp(method, "baseline") == 0) {
            dist = ssd_distance(target_features, database_features[i]);
        } else if (strcmp(method, "histogram") == 0) {
            dist = histogram_intersection(target_features, database_features[i]);
        } else if (strcmp(method, "multi_histogram") == 0) {
            dist = multi_histogram_distance(target_features, database_features[i]);
        } else if (strcmp(method, "texture_color") == 0) {
            dist = texture_color_distance(target_features, database_features[i], bins, 16);
        } else {
            dist = -1.0f;
        }
        
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
        printf("%-30s %.4f\n", distances[i].filename, distances[i].distance);
    }
    
    return 0;
}