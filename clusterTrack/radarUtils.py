"""
Radar Utilities Module

This module contains utility functions for radar data processing, including
filtering radar points based on geometric constraints and sensor configurations.
"""

import numpy as np
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from typing import Tuple
from config import FILTERING_PARAMS


def get_radar_points_within_lidar_radius(all_radar_pcs, nusc, sample, radius=FILTERING_PARAMS['lidar_radius']):
    """
    Get all radar points within a specified radius of the lidar position in global coordinates.
    
    This function filters radar points to only include those within a certain distance
    of the lidar sensor, which can be useful for maintaining consistency between
    different sensor modalities.
    
    Args:
        all_radar_pcs: RadarPointCloud object with points in global coordinates
        nusc: NuScenes instance
        sample: Current sample record
        radius: Radius in meters (default: 50m)
    
    Returns:
        tuple: A tuple containing:
            - filtered_points: Radar points within the radius
            - mask: Boolean mask indicating which points are within radius
            - lidar_global_pos: Lidar position in global coordinates
            - distances: Distances from each radar point to lidar position
    """
    
    # Lidar sensor data and transforms
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    
    # Ego pose (car position) at lidar timestamp
    ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    
    # Lidar sensor calibration (position relative to car)
    lidar_cs = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    
    # Transform lidar sensor position to global coordinates
    # Step 1: Car position in global coordinates
    global_from_car = transform_matrix(
        ego_pose['translation'], 
        Quaternion(ego_pose['rotation']), 
        inverse=False
    )
    
    # Step 2: Lidar position relative to car
    car_from_lidar = transform_matrix(
        lidar_cs['translation'], 
        Quaternion(lidar_cs['rotation']), 
        inverse=False
    )
    
    # Step 3: Combine transforms to get lidar position in global coordinates
    global_from_lidar = np.dot(global_from_car, car_from_lidar)
    
    # Lidar position in global coordinates (x, y, z)
    lidar_global_pos = global_from_lidar[:3, 3]
    
    # First 3 dimensions are x, y, z in global coordinates of radar point clouds
    radar_xyz = all_radar_pcs.points[:3, :]  # Shape: (3, N)
    
    # Distances from each radar point to lidar position
    lidar_pos_reshaped = lidar_global_pos.reshape(3, 1)  # Shape: (3, 1)
    distances = np.sqrt(np.sum((radar_xyz - lidar_pos_reshaped)**2, axis=0))
    mask = distances <= radius
    
    # Filter radar points based on their distance to lidar
    filtered_points = all_radar_pcs.points[:, mask]
    
    return filtered_points, mask, lidar_global_pos, distances


def extract_radar_features(filtered_radar_points):
    """
    Extract common features from filtered radar points.
    
    Args:
        filtered_radar_points: Filtered radar point cloud data
        
    Returns:
        dict: Dictionary containing extracted features
    """
    features = {}
    
    features['x'] = filtered_radar_points[0, :]
    features['y'] = filtered_radar_points[1, :]
    features['z'] = filtered_radar_points[2, :]
    
    features['vx'] = filtered_radar_points[7, :]
    features['vy'] = filtered_radar_points[8, :]
    features['v'] = np.sqrt(features['vx']**2 + features['vy']**2)
    
    if len(features['v']) > 0 and np.max(features['v']) != np.min(features['v']):
        features['normalized_v'] = (features['v'] - np.min(features['v'])) / (np.max(features['v']) - np.min(features['v']))
        features['normalized_v_mean'] = np.mean(features['normalized_v'])
    else:
        features['normalized_v'] = np.zeros_like(features['v'])
        features['normalized_v_mean'] = 0.0
    
    features['dyn_prop'] = filtered_radar_points[3, :]
    
    features['rcs'] = filtered_radar_points[5, :]
    
    if len(features['rcs']) > 0 and np.max(features['rcs']) != np.min(features['rcs']):
        features['normalized_rcs'] = (features['rcs'] - np.min(features['rcs'])) / (np.max(features['rcs']) - np.min(features['rcs']))
        features['normalized_rcs_median'] = np.median(features['normalized_rcs'])
    else:
        features['normalized_rcs'] = np.zeros_like(features['rcs'])
        features['normalized_rcs_median'] = 0.0
    
    if filtered_radar_points.shape[0] > 21:
        features['weighted_rcs'] = filtered_radar_points[21, :]
        features['all_distance'] = filtered_radar_points[20, :]
        features['far_distance'] = np.max(features['all_distance']) if len(features['all_distance']) > 0 else 0.0
        features['closest_distance'] = np.min(features['all_distance']) if len(features['all_distance']) > 0 else 0.0
        features['min_rcs_weighted'] = np.min(features['weighted_rcs']) if len(features['weighted_rcs']) > 0 else 0.0
        features['max_rcs_weighted'] = np.max(features['weighted_rcs']) if len(features['weighted_rcs']) > 0 else 0.0
    
    return features


def filter_radar_by_dynamic_properties(radar_points, valid_dyn_props=None):
    """
    Filter radar points based on dynamic properties.
    
    Args:
        radar_points: Radar point cloud data
        valid_dyn_props: List of valid dynamic property values to keep
        
    Returns:
        Filtered radar points
    """
    if valid_dyn_props is None:
        return radar_points
    
    dyn_prop = radar_points[3, :]
    mask = np.isin(dyn_prop, valid_dyn_props)
    return radar_points[:, mask]


def compute_radar_statistics(radar_points):
    """
    Compute basic statistics for radar point cloud.
    
    Args:
        radar_points: Radar point cloud data
        
    Returns:
        dict: Dictionary containing radar statistics
    """
    stats = {}
    
    if radar_points.shape[1] == 0:
        return {"num_points": 0}
    
    stats['num_points'] = radar_points.shape[1]
    
    xyz = radar_points[:3, :]
    stats['spatial_extent'] = {
        'x_range': (np.min(xyz[0, :]), np.max(xyz[0, :])),
        'y_range': (np.min(xyz[1, :]), np.max(xyz[1, :])),
        'z_range': (np.min(xyz[2, :]), np.max(xyz[2, :]))
    }
    
    if radar_points.shape[0] > 8:
        vx, vy = radar_points[7, :], radar_points[8, :]
        v_magnitude = np.sqrt(vx**2 + vy**2)
        stats['velocity'] = {
            'mean_speed': np.mean(v_magnitude),
            'max_speed': np.max(v_magnitude),
            'speed_std': np.std(v_magnitude)
        }
    
    if radar_points.shape[0] > 5:
        rcs = radar_points[5, :]
        stats['rcs'] = {
            'mean_rcs': np.mean(rcs),
            'median_rcs': np.median(rcs),
            'rcs_std': np.std(rcs)
        }
    
    return stats 