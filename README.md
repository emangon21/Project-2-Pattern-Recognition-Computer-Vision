# Content-Based Image Retrieval - Project 2

CS 5330 Pattern Recognition and Computer Vision - Project 2

## Time Travel Days
**Time travel days used:** 0

## Team Members
- Eugenie Mangon - Setup + Tasks 1-4 + corresponding report
- Ishika Gupta - Tasks 5-7 + Extensions + corresponding report
 
## Project Description
Implementation of a content-based image retrieval (CBIR) system using multiple feature extraction methods:
- Baseline matching (7x7 center square with SSD)
- Color histogram matching (RGB histogram with histogram intersection)
- Multi-histogram matching with spatial layout (top/bottom halves)
- Texture and color features (Sobel gradient magnitude + RGB histogram)
- Deep network embeddings (ResNet18)

## Setup

### Prerequisites
- C++11 compiler (clang++)
- OpenCV 4.x
- macOS with Homebrew

### Compilation
```bash
# Clean previous builds
make clean

# Compile all programs
make all
```

## Project Structure
```
Project-2/
├── bin/                      # Compiled executables
│   ├── baseline              # Task 1: Baseline feature extraction
│   ├── histogram             # Task 2: Color histogram extraction
│   ├── multi_histogram       # Task 3: Multi-histogram extraction
│   ├── texture_color         # Task 4: Texture+color extraction
│   └── query                 # Query program for all methods
├── data/                     # Image database (not tracked in git)
│   └── olympus/              # Olympus image dataset (1106 images)
├── include/                  # Header files
│   ├── csv_util.h            # CSV reading/writing utilities
│   └── features.h            # Feature extraction functions
├── src/                      # Source files
│   ├── baseline.cpp          # Task 1 implementation
│   ├── histogram.cpp         # Task 2 implementation
│   ├── multi_histogram.cpp   # Task 3 implementation
│   ├── texture_color.cpp     # Task 4 implementation
│   ├── query.cpp             # Query program
│   ├── features.cpp          # Feature extraction implementations
│   └── csv_util.cpp          # CSV utilities
├── Makefile                  # Build configuration
└── README.md                 # This file
```

## Task 1: Baseline Matching

**Method:** 7×7 center square (147 features)  
**Distance Metric:** Sum of Squared Differences (SSD)

### Extract Features
```bash
./bin/baseline ./data/olympus features_baseline.csv
```

### Query for Similar Images
```bash
./bin/query ./data/olympus/pic.1016.jpg features_baseline.csv baseline 4
```

---

## Task 2: Histogram Matching

**Method:** RGB color histogram (8 bins per channel = 512 bins)  
**Distance Metric:** Histogram intersection

### Extract Features
```bash
./bin/histogram ./data/olympus features_histogram.csv 8
```

### Query for Similar Images
```bash
./bin/query ./data/olympus/pic.0164.jpg features_histogram.csv histogram 4 8
```

---

## Task 3: Multi-histogram Matching

**Method:** Spatial RGB histograms (top + bottom halves, 1024 features)  
**Distance Metric:** Weighted histogram intersection (0.5 each)

### Extract Features
```bash
./bin/multi_histogram ./data/olympus features_multi_histogram.csv 8
```

### Query for Similar Images
```bash
./bin/query ./data/olympus/pic.0274.jpg features_multi_histogram.csv multi_histogram 4 8
```

---

## Task 4: Texture and Color

**Method:** RGB histogram (512) + Sobel magnitude histogram (16) = 528 features  
**Distance Metric:** Weighted histogram intersection (0.5 each)

### Extract Features
```bash
./bin/texture_color ./data/olympus features_texture_color.csv 8 16
```

### Query for Similar Images
```bash
./bin/query ./data/olympus/pic.0535.jpg features_texture_color.csv texture_color 4 8
```

### Compare with Tasks 2 and 3
```bash
./bin/query ./data/olympus/pic.0535.jpg features_histogram.csv histogram 4 8
./bin/query ./data/olympus/pic.0535.jpg features_multi_histogram.csv multi_histogram 4 8
```

---

## Task 5: Deep Network Embeddings

**Method:** ResNet18 embeddings (512-dimensional feature vectors)  
**Distance Metric:** Cosine distance or SSD

*[To be implemented]*

---

## Task 6: Compare DNN vs Classic Features

*[To be implemented]*

---

## Task 7: Custom Design

*[To be implemented]*

---

## Extension

*[To be implemented]*

---

## General Query Usage
```bash
./bin/query <target_image> <feature_csv> <method> <N> [bins]
```

**Parameters:**
- `target_image`: Path to query image
- `feature_csv`: Pre-computed feature database
- `method`: `baseline`, `histogram`, `multi_histogram`, or `texture_color`
- `N`: Number of top matches to return
- `bins`: (Optional) Number of bins per channel (default: 8)

**Example:**
```bash
./bin/query ./data/olympus/pic.0164.jpg features_histogram.csv histogram 3 8
```

---

## Operating System
macOS (Apple Silicon)

## IDE
Visual Studio Code
