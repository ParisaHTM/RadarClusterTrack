# config.py

CAMS = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_FRONT_LEFT'
]

CAMERA_LABELS = {
    'CAM_FRONT_LEFT': 'FRONT LEFT',
    'CAM_FRONT': 'FRONT',
    'CAM_FRONT_RIGHT': 'FRONT RIGHT',
    'CAM_BACK_LEFT': 'BACK LEFT',
    'CAM_BACK': 'BACK',
    'CAM_BACK_RIGHT': 'BACK RIGHT'
}

DEFAULT_PATHS = {
    "data_path": "C:/Users/qxy699/Documents/GAFusion/nuscences/nuScenes-lidarseg-mini-v1.0",
    "data_dir": "./clustering_data_pkl_rcs",
    "individual_frames_root": "./360_cam_individual_frames",
    "save_root": "./360_cam_videos",
    "combined_frames_root": "./360_cam_combined_frames",
    "global_map_root": "./global_map_videos"
}

VIDEO_PARAMS = {
    "fps": 5,
    "frame_size": (1600, 900)  # width, height
}
