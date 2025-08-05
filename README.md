# Radar Cluster Track - 360 Camera Visualization System

This project provides a comprehensive pipeline for radar point cloud clustering, tracking, visualization, and precision evaluation using the NuScenes dataset. The system processes radar data to identify and track clusters across multiple frames and cameras, visualizes results on both 360-degree camera views and global maps, and evaluates clustering precision.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [1. Clustering and Tracking (clusterTrack)](#1-clustering-and-tracking-clustertrack)
- [2. Visualization (visCamMap)](#2-visualization-viscammap)
- [3. Precision Evaluation (precision)](#3-precision-evaluation-precision)
- [Output Structure](#output-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Overview

The project consists of three main components:

1. **Clustering and Tracking** (`clusterTrack/`) - Processes radar data to identify and track clusters
2. **Visualization** (`visCamMap/`) - Creates visual outputs for camera views and global maps
3. **Precision Evaluation** (`precision/`) - Evaluates clustering accuracy and generates performance metrics
![Method](https://github.com/ParisaHTM/RadarClusterTrack/blob/main/figs/Method.png)
## Visual Results
Our model successfully detects and tracks the radar cluster corresponding to the bus as a moving object over a long sequence of frames.

![Video](https://github.com/ParisaHTM/RadarClusterTrack/blob/main/figs/scene_0014.gif)
## Installation

### Prerequisites
- Python 3.8+
- NuScenes dataset (v1.0-mini or full version)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Dataset Setup
Download the NuScenes dataset and update the `data_path` in configuration files or use command-line arguments to specify the path.

## Quick Start

1. **Process radar data and generate clusters:**
   ```bash
   cd clusterTrack
   python processScenes.py --data-path /path/to/nuscenes --scene-name scene-0061
   ```

2. **Visualize on 360-degree cameras:**
   ```bash
   cd ../visCamMap
   python main360.py --mode single --scene scene-0061
   ```

3. **Visualize on global map:**
   ```bash
   python mainMap.py --mode single --scene scene-0061
   ```

4. **Evaluate precision:**
   ```bash
   cd ../precision
   python main.py --data-path /path/to/nuscenes --plot-type precisionSimga5
   ```

---

## 1. Clustering and Tracking (clusterTrack)

**Step 1: Navigate to the clustering directory**
```bash
cd clusterTrack
```

### Main Scripts and Usage

#### `processScenes.py` - Main Entry Point
*Processes radar data from NuScenes dataset and generates clustering results.*

**Usage:**
```bash
# Process a single scene
python processScenes.py --data-path /path/to/nuscenes --scene-name scene-0061

# Process a range of scenes
python processScenes.py --data-path /path/to/nuscenes --start 0 --end 5

# Process all scenes with custom output directory
python processScenes.py --data-path /path/to/nuscenes --output-dir ./custom_output
```

**Parameters:**
- `--data-path`: Path to NuScenes dataset (required)
- `--scene-name`: Process specific scene by name
- `--start`: Starting scene index (default: 0)
- `--end`: Ending scene index (default: all remaining)
- `--output-dir`: Custom output directory for results

---

## 2. Visualization (visCamMap)

**Step 2: Navigate to the visualization directory**
```bash
cd visCamMap
```

### Main Scripts and Usage

#### `main360.py` - 360-Degree Camera Visualization
*Creates videos showing radar clusters overlaid on 360-degree camera views.*

**Usage:**
```bash
# Visualize single scene
python main360.py --mode single --scene scene-0061

# Visualize range of scenes
python main360.py --mode range --start 0 --end 3

# Visualize all available scenes
python main360.py --mode all
```

**Output:** Creates individual camera frame images, combined panoramic frames, and MP4 videos.

#### `mainMap.py` - Global Map Visualization
*Creates videos showing radar clusters on global map with road context.*

**Usage:**
```bash
# Visualize single scene on global map
python mainMap.py --mode single --scene scene-0061

# Visualize multiple scenes
python mainMap.py --mode range --start 0 --end 5

# Process all scenes
python mainMap.py --mode all
```

**Output:** Creates global map visualizations with cluster tracking and off-road detection.

---

## 3. Precision Evaluation (precision)

**Step 3: Navigate to the precision evaluation directory**
```bash
cd precision
```

### Main Scripts and Usage

#### `main.py` - Evaluation Entry Point
*Comprehensive precision evaluation with multiple plot types.*

**Usage:**
```bash
# Basic precision evaluation
python main.py --data-path /path/to/nuscenes --plot-type precisionSimga5

# Cluster count analysis
python main.py --data-path /path/to/nuscenes --plot-type num_cluster

# Movement pattern analysis
python main.py --data-path /path/to/nuscenes --plot-type mov_stationary_cluster

# RCS-based analysis
python main.py --data-path /path/to/nuscenes --plot-type rcs_cluster
```

**Available Plot Types:**
- `precisionSimga5`: Accuracy vs consecutive frame requirements
- `num_cluster`: Total cluster count analysis
- `mov_stationary_cluster`: Movement pattern categorization
- `percentage_mov_status_cluster`: Movement pattern percentages
- `precistion_mov_status_cluster`: Movement pattern precision scores
- `rcs_cluster`: RCS-based cluster analysis
- `percentage_rcs_status_cluster`: RCS pattern percentages
- `precision_rcs_status_cluster`: RCS pattern precision scores

**Parameters:**
- `--sigma`: Similarity threshold (default: 5)
- `--range-sigma`: List of sigma values for comparison
- `--consecutive-frames`: Range of frame requirements (default: 1-30)
- `--version`: NuScenes dataset version (default: v1.0-mini)

### Core Modules

#### `precisionSimga5.py` - Precision Analysis
*Implements precision evaluation and visualization functions.*

**Key Functions:**
- `precisionSimga5()`: Main evaluation function
  - Processes all scenes with specified consecutive frame requirements
  - Calculates accuracy and standard deviation for each frame requirement
  - Generates accuracy vs frequency plot
  - Returns comprehensive results dictionary

- `plot_num_cluster()`: Analyzes total cluster counts
  - Creates bar chart of cluster counts vs frame requirements
  - Shows relationship between frame persistence and cluster detection

- `plot_mov_stationary_cluster()`: Movement pattern analysis
  - Categorizes clusters by movement behavior
  - Tracks precision scores for different motion types
  - Generates movement pattern distribution plots

**Movement Categories:**
- Moving, Stationary, Oncoming, Cross-moving, Cross-stationary, Stopped, Unknown

---

## Output Structure

The system generates several types of outputs:

```
project_root/
├── clustering_data_pkl_rcs/          # Clustering results (pickle files)
│   ├── scene-0061_clustering_data.pkl
│   └── scene-0103_clustering_data.pkl
├── 360_cam_videos/                   # 360-degree camera videos
│   ├── scene-0061.mp4
│   └── scene-0103.mp4
├── 360_cam_individual_frames/        # Individual camera frames
│   └── scene-0061/
│       ├── front/
│       ├── front_left/
│       └── ...
├── 360_cam_combined_frames/          # Combined panoramic frames
│   └── scene-0061/
│       ├── frame_0000.png
│       └── frame_0001.png
├── global_map_videos/                # Global map visualizations
│   ├── scene-0061_cluster_map.mp4
│   └── scene-0061/
│       ├── frame_0000.png
│       └── frame_0001.png
├── accuracy_vs_Freq_rcs_dyna.png    # Precision evaluation plots
└── num_cluster.png
```

### File Descriptions

**Clustering Results (`clustering_data_pkl_rcs/`):**
- Binary pickle files containing processed clustering data
- Each file contains frame-by-frame cluster information
- Includes cluster positions, IDs, RCS values, and tracking data

**360-Degree Visualizations:**
- `360_cam_videos/`: MP4 videos showing radar clusters on camera views
- `360_cam_individual_frames/`: Separate images for each camera view
- `360_cam_combined_frames/`: Panoramic view combining all 6 cameras

**Global Map Visualizations:**
- `global_map_videos/`: MP4 videos showing clusters on road maps
- Individual frame images with road context and off-road detection
- Cluster movement analysis and semantic labeling

**Evaluation Results:**
- Various PNG plots showing accuracy, cluster counts, and performance metrics

---

## Configuration

### Path Configuration
Update paths in configuration files or use command-line arguments:

**For `clusterTrack/config.py`:**
```python
DEFAULT_PATHS = {
    'pickle_save_root': './clustering_data_pkl_rcs',
}
```

**For `visCamMap/config.py`:**
```python
DEFAULT_PATHS = {
    "data_path": "/path/to/your/nuscenes/dataset",
    "data_dir": "./clustering_data_pkl_rcs",
    # ... other paths
}
```

### Processing Parameters

**Clustering Parameters:**
```python
CLUSTERING_PARAMS = {
    'velocity_threshold': 0.5,      # m/s
    'position_threshold': 2.0,      # meters
    'min_cluster_size': 3,          # minimum points per cluster
}
```

**Tracking Parameters:**
```python
TRACKING_PARAMS = {
    'max_distance_threshold': 5.0,  # meters
    'cross_camera_threshold': 3.0,  # meters
    'max_cluster_age': 10,          # frames
    'max_colors': 50                # color palette size
}
```

### Video Output Settings
```python
VIDEO_PARAMS = {
    "fps": 5,
    "frame_size": (1600, 900)  # width, height
}
```

---

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure you're in the correct directory
cd clusterTrack  # or visCamMap, precision
# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**2. Missing NuScenes Data**
- Verify dataset path in config files
- Ensure you have the correct NuScenes version (v1.0-mini or full)
- Check that the dataset includes radar data

**3. Memory Issues**
- Process scenes individually using `--scene-name` parameter
- Reduce the number of concurrent scenes being processed
- Consider using a machine with more RAM for large datasets

**4. Visualization Issues**
- Ensure clustering data exists before running visualization
- Check that pickle files are properly generated in `clustering_data_pkl_rcs/`
- Verify output directories have write permissions

**5. Performance Optimization**
- Use `--start` and `--end` parameters to process scenes in batches
- Adjust clustering parameters to reduce computational load

### File Dependencies

Make sure to run scripts in the correct order:
1. First: `clusterTrack/processScenes.py` (generates clustering data)
2. Then: `visCamMap/main360.py` or `mainMap.py` (creates visualizations) OR
3. Finally: `precision/main.py` (evaluates results)


## Contact
For questions, issues, or contributions:
- Create an issue on GitHub
- Contact: qxu699@mocs.utc.edu / parisahatami001@gmail.com
---

*Last updated: 2024*
