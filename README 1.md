# Content-Based Image Retrieval - Project 2

CS5760 Pattern Recognition - Project 2

## Team Members
- Eugenie Mangon - Setup + Tasks 1-4
- Ishika Gupta - Tasks 5-7 + Extension

## Time Travel Days
**Time travel days used:** 1

## Project Description
Implementation of a content-based image retrieval (CBIR) system using multiple feature extraction methods:
- Baseline matching (7x7 center square with SSD)
- Color histogram matching (RGB histogram with histogram intersection)
- Multi-histogram matching with spatial layout (top/bottom halves)
- Texture and color features (Sobel gradient magnitude + RGB histogram)
- Deep network embeddings (ResNet18)

## Setup

### Prerequisites
- C++17 compiler (clang++)
- OpenCV 4.x
- macOS with Homebrew (Apple Silicon)

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
│   ├── custom                # Task 7: Custom feature extraction
│   ├── banana                # Extension: Banana detection
│   ├── bluebin               # Extension: Blue bin detection
│   ├── face                  # Extension: Face detection
│   ├── gui                   # Extension: GUI application
│   └── query                 # Query program for all methods
├── data/                     # Image database
│   ├── olympus/              # Olympus image dataset (1106 images)
│   └── resnet18_embeddings.csv  # Task 5: Pre-computed ResNet18 features
├── include/                  # Header files
│   ├── csv_util.h            # CSV reading/writing utilities
│   └── features.h            # Feature extraction functions
├── src/                      # Source files
│   ├── baseline.cpp          # Task 1 implementation
│   ├── histogram.cpp         # Task 2 implementation
│   ├── multi_histogram.cpp   # Task 3 implementation
│   ├── texture_color.cpp     # Task 4 implementation
│   ├── custom.cpp            # Task 7 implementation
│   ├── banana.cpp            # Extension: Banana detection
│   ├── bluebin.cpp           # Extension: Blue bin detection
│   ├── face.cpp              # Extension: Face detection
│   ├── gui.cpp               # Extension: GUI application
│   ├── query.cpp             # Query program
│   ├── features.cpp          # Feature extraction implementations
│   └── csv_util.cpp          # CSV utilities
├── results/                  # Query results and analysis
├── Makefile                  # Build configuration
└── README.md                 # This file
```

---

## Task 1: Baseline Matching

**Method:** 7×7 center square (147 features)  
**Distance Metric:** Sum of Squared Differences (SSD)

### Compilation
```bash
cd src
make clean
make baseline
cd ../bin
```

### Extract Features
```bash
./baseline ../data/olympus features_baseline.csv
```

### Query for Similar Images
```bash
./query ../data/olympus/pic.1016.jpg features_baseline.csv baseline 4
```

---

## Task 2: Histogram Matching

**Method:** RGB color histogram (8 bins per channel = 512 bins)  
**Distance Metric:** Histogram intersection

### Compilation
```bash
cd src
make clean
make histogram
cd ../bin
```

### Extract Features
```bash
./histogram ../data/olympus features_histogram.csv 8
```

### Query for Similar Images
```bash
./query ../data/olympus/pic.0164.jpg features_histogram.csv histogram 4 8
```

---

## Task 3: Multi-histogram Matching

**Method:** Spatial RGB histograms (top + bottom halves, 1024 features)  
**Distance Metric:** Weighted histogram intersection - 0.5 each

### Compilation
```bash
cd src
make clean
make multi_histogram
cd ../bin
```

### Extract Features
```bash
./multi_histogram ../data/olympus features_multi_histogram.csv 8
```

### Query for Similar Images
```bash
./query ../data/olympus/pic.0274.jpg features_multi_histogram.csv multi_histogram 4 8
```

---

## Task 4: Texture and Color

**Method:** RGB histogram (512) + Sobel magnitude histogram (16) = 528 features  
**Distance Metric:** Weighted histogram intersection - 0.5 each

### Compilation
```bash
cd src
make clean
make texture_color
cd ../bin
```

### Extract Features
```bash
./texture_color ../data/olympus features_texture_color.csv 8 16
```

### Query for Similar Images
```bash
./query ../data/olympus/pic.0535.jpg features_texture_color.csv texture_color 4 8
```

### Compare with Tasks 2 and 3
```bash
./query ../data/olympus/pic.0535.jpg features_histogram.csv histogram 4 8
./query ../data/olympus/pic.0535.jpg features_multi_histogram.csv multi_histogram 4 8
```

---

## Task 5: Deep Network Embeddings

**Method:** ResNet18 embeddings (512-dimensional feature vectors)  
**Distance Metric:** Cosine distance or SSD

### Compilation
```bash
cd src
make clean
make query
cd ../bin
```

### Query for Similar Images
```bash
./query ../data/olympus/pic.XXXX.jpg ../data/resnet18_embeddings.csv dnn 4
```

**Note:** ResNet18 features are pre-computed and stored in `data/resnet18_embeddings.csv`. No separate feature extraction program is needed for this task.

---

## Task 6: Compare DNN vs Classic Features

**Objective:** Compare the performance of deep network embeddings (Task 5) against classical feature extraction methods (Tasks 1-4).

### Running Comparisons
Use the query program with different feature files to compare results:
```bash
# Compare same target image across all methods
./query ../data/olympus/pic.XXXX.jpg features_baseline.csv baseline 10
./query ../data/olympus/pic.XXXX.jpg features_histogram.csv histogram 10 8
./query ../data/olympus/pic.XXXX.jpg features_multi_histogram.csv multi_histogram 10 8
./query ../data/olympus/pic.XXXX.jpg features_texture_color.csv texture_color 10 8
./query ../data/olympus/pic.XXXX.jpg ../data/resnet18_embeddings.csv dnn 10
```

---

## Task 7: Custom Design

**Method:** Custom feature extraction combining multiple approaches  
**Distance Metric:** Custom distance metric

### Compilation
```bash
cd src
make clean
make custom
cd ../bin
```

### Extract Features
```bash
./custom ../data/olympus features_custom.csv
```

### Query for Similar Images
```bash
./query ../data/olympus/pic.XXXX.jpg features_custom.csv custom 4
```

---

## Extension

### Banana Detection
**Method:** Custom feature extraction optimized for banana identification

#### Compilation
```bash
cd src
make clean
make banana
cd ../bin
```

#### Extract Features
```bash
./banana ../data/olympus features_banana.csv
```

#### Query
```bash
./query ../data/olympus/pic.XXXX.jpg features_banana.csv banana 4
```

---

### Blue Bin Detection
**Method:** Custom feature extraction optimized for blue recycling bin identification

#### Compilation
```bash
cd src
make clean
make bluebin
cd ../bin
```

#### Extract Features
```bash
./bluebin ../data/olympus features_bluebin.csv
```

#### Query
```bash
./query ../data/olympus/pic.XXXX.jpg features_bluebin.csv bluebin 4
```

---

### Face Detection
**Method:** Custom feature extraction optimized for face identification

#### Compilation
```bash
cd src
make clean
make face
cd ../bin
```

#### Extract Features
```bash
./face ../data/olympus features_face.csv
```

#### Query
```bash
./query ../data/olympus/pic.XXXX.jpg features_face.csv face 4
```

---

### GUI Application
**Interactive visual interface for image retrieval**

#### Compilation
```bash
cd src
make clean
make gui
cd ../bin
```

#### Run GUI
```bash
./gui
```

---

## General Query Usage
```bash
./query <target_image> <feature_csv> <method> <N> [bins]
```

**Parameters:**
- `target_image`: Path to query image
- `feature_csv`: Pre-computed feature database
- `method`: `baseline`, `histogram`, `multi_histogram`, `texture_color`, `custom`, `dnn`, `banana`, `bluebin`, or `face`
- `N`: Number of top matches to return
- `bins`: (Optional) Number of bins per channel (default: 8, only for histogram-based methods)

**Examples:**
```bash
./query ../data/olympus/pic.0164.jpg features_histogram.csv histogram 3 8
./query ../data/olympus/pic.1016.jpg features_baseline.csv baseline 5
./query ../data/olympus/pic.0535.jpg ../data/resnet18_embeddings.csv dnn 10
```

---

## Operating System
macOS (Apple Silicon)

## IDE
Visual Studio Code

---

## Notes
- All feature extraction programs process the entire olympus dataset (1106 images)
- Feature CSV files are saved in the project root directory
- Query program works with any pre-computed feature file
- ResNet18 embeddings are pre-computed and do not require a separate extraction step
- Extension programs (banana, bluebin, face) use specialized feature extraction for specific object detection tasks