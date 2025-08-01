"""
Configuration Module

This module contains all configuration constants, mappings, and settings
used throughout the radar data processing pipeline.
"""

# Camera configuration
CAMS = [
    'CAM_FRONT', 
    'CAM_FRONT_RIGHT', 
    'CAM_BACK_RIGHT', 
    'CAM_BACK', 
    'CAM_BACK_LEFT',
    'CAM_FRONT_LEFT'
]

RADARS_ID = {
    "RADAR_FRONT": 1,
    "RADAR_FRONT_LEFT": 2,
    "RADAR_FRONT_RIGHT": 3,
    "RADAR_BACK": 4,
    "RADAR_BACK_LEFT": 5,
    "RADAR_BACK_RIGHT": 6,
}

# Radar sensors associated with each camera view
RADARS_FOR_CAMERA = {
    'CAM_FRONT_LEFT': ["RADAR_FRONT_LEFT", "RADAR_FRONT"],
    'CAM_FRONT_RIGHT': ["RADAR_FRONT_RIGHT", "RADAR_FRONT"],
    'CAM_FRONT': ["RADAR_FRONT_RIGHT", "RADAR_FRONT_LEFT", "RADAR_FRONT"],
    'CAM_BACK_LEFT': ["RADAR_BACK_LEFT", "RADAR_FRONT_LEFT"],
    'CAM_BACK_RIGHT': ["RADAR_BACK_RIGHT", "RADAR_FRONT_RIGHT"],
    'CAM_BACK': ["RADAR_BACK_RIGHT", "RADAR_BACK_LEFT"]
}


# Default radar processing parameters
RADAR_PROCESSING_PARAMS = {
    'invalid_states': [0, 4, 8, 9, 10, 11, 12, 15, 16, 17],
    'dynoprop_state': range(8),
    'ambig_state': [3, 4],
    'pdh0': range(2, 8),
    'min_distance': 1.0,
    'rcs_threshold': 4.0
}

# Default clustering parameters
CLUSTERING_PARAMS = {
    'velocity_threshold': 0.5,
    'position_threshold': 2.0,
    'min_cluster_size': 3,
    'max_cluster_size': None
}

# Default tracking parameters
TRACKING_PARAMS = {
    'max_distance_threshold': 5.0,
    'cross_camera_threshold': 3.0,
    'max_cluster_age': 10,
    'max_colors': 50
}


# Default filtering parameters
FILTERING_PARAMS = {
    'lidar_radius': 50.0,  # meters
    'merge_threshold': 5.0,  # meters for cluster merging
    'stability_threshold': 3.0  # meters for cluster stability analysis
}

# File paths
DEFAULT_PATHS = {
    'pickle_save_root': './clustering_data_pkl_rcs',
}

# NuScenes dataset configuration
NUSCENES_CONFIG = {
    'version': 'v1.0-mini',
    'verbose': True
}

# Processing flags
PROCESSING_FLAGS = {
    'save_pickle': True,
}
