"""
Radar Point Cloud Processing Module
A modified version of the NuScenes RadarPointCloud class (https://github.com/nutonomy/nuscenes-devkit/blob/932064c611ddfe06e3d1fadea904eb365482a03b/python-sdk/nuscenes/utils/data_classes.py)
This module contains utilities for processing radar point clouds with RCS (Radar Cross Section)
calculations and distance computations for NuScenes dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from nuscenes.nuscenes import NuScenes 
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from functools import reduce
from typing import Dict, Tuple, List
import os.path as osp
import numpy as np
from pyquaternion import Quaternion
from visCamMap.config import RADAR_PROCESSING_PARAMS, RADARS_ID


def transform_rcs(points, threshold=4):
    """
    Transform radar point cloud values based on its sign.
    Negative values -> small positive (0-0.2)
    Positive values -> larger positive (0.2-1)
    
    Args:
        points: RCS values to transform
        threshold: Threshold for RCS transformation (default: 4)
    
    Returns:
        Transformed RCS values
    """
    transformed = np.empty_like(points, dtype=np.float32)
    
    # Boolean masks
    negative_mask = points <= 0
    positive_mask = points > 0

    # Different sigmoid curves for each mask
    transformed[negative_mask] = 1 / (1 + np.exp(-(points[negative_mask] - 2) / (threshold)))
    transformed[positive_mask] = points[positive_mask] * (1 / (1 + np.exp(-(points[positive_mask] - 2) / threshold)))
    return transformed


def compute_rcs_area(rcs_values, threshold=4):
    """
    Compute pi * rcs_transformed^2 for RCS values.
    
    Args:
        rcs_values: Array of RCS values
        threshold: Threshold for RCS transformation (default: 4)
    
    Returns:
        Array of RCS areas (pi * rcs_transformed^2)
    """
    rcs_transformed = transform_rcs(rcs_values, threshold)
    rcs_areas = np.pi * rcs_transformed ** 2
    return rcs_areas.astype(np.float32)


class RadarPointCloudWithRcsDistance(RadarPointCloud):
    """
    Extended RadarPointCloud class that includes RCS area and distance calculations.
    
    This class extends the NuScenes RadarPointCloud to include additional dimensions:
    - Distance to ego vehicle
    - RCS area calculations
    - Point labeling
    """
    
    RADARS_ID = RADARS_ID

    @staticmethod
    def nbr_dims() -> int:
        """
        Returns the number of dimensions.
        
        Returns:
            Number of dimensions (23 total):
            - Original RadarPointCloud dimensions
            - Point ID
            - Radar channel ID  
            - Distance to ego
            - RCS area
            - Label value (0: off road, 1: car_moving, 2: car_stationary, 
                          3: ped_crossing, 4: ped_stationary, 5: barrier)
        """
        return 23
    
    @staticmethod
    def compute_distance_to_ego(points: np.ndarray, ego_translation: List[float]) -> np.ndarray:
        """
        Compute the Euclidean distance from each point to the ego vehicle.
        
        Args:
            points: Point cloud with shape (n_dims, n_points) where first 3 dimensions are x, y, z
            ego_translation: Ego vehicle translation [x, y, z]
        
        Returns:
            Array of distances with shape (1, n_points)
        """
        xyz = points[:3, :]
        ego = np.array(ego_translation).reshape(3, 1)
        offset = xyz - ego
        distances = np.sqrt(np.sum(offset**2, axis=0, keepdims=True))
        return distances.astype(np.float32)
    
    @staticmethod
    def compute_rcs_area_for_points(rcs_values: np.ndarray, threshold: float = 4) -> np.ndarray:
        """
        Compute RCS area (pi * rcs_transformed^2) for each point.
        
        Args:
            rcs_values: RCS values for points (dimension 5 in radar point cloud)
            threshold: Threshold for RCS transformation (default: 4)
        
        Returns:
            Array of RCS areas with shape (1, n_points)
        """
        rcs_areas = compute_rcs_area(rcs_values, threshold)
        return rcs_areas.reshape(1, -1).astype(np.float32)
         
    @classmethod
    def from_file_multisweep(cls, 
                            nusc: 'NuScenes', 
                            sample_rec: Dict, 
                            chan: str, 
                            invalid_states = RADAR_PROCESSING_PARAMS['invalid_states'],
                            dynoprop_state = RADAR_PROCESSING_PARAMS['dynoprop_state'],
                            ambig_state = RADAR_PROCESSING_PARAMS['ambig_state'],
                            pdh0 = RADAR_PROCESSING_PARAMS['pdh0'],
                            min_distance: float = RADAR_PROCESSING_PARAMS['min_distance'],
                            rcs_threshold: float = RADAR_PROCESSING_PARAMS['rcs_threshold']) -> Tuple['RadarPointCloudWithRcsDistance', np.ndarray]:
        """
        Return a point cloud that aggregates multiple sweeps with enhanced features.
        
        As every sweep is in a different coordinate frame, we need to map the coordinates 
        to a single reference frame. As every sweep has a different timestamp, we need to 
        account for that in the transformations and timestamps.
        
        Args:
            nusc: A NuScenes instance
            sample_rec: The current sample
            chan: The radar channel from which we track back n sweeps to aggregate the point cloud
            invalid_states: States to filter out
            dynoprop_state: Dynamic property states to include
            ambig_state: Ambiguous states to include
            pdh0: False alarm filter range
            min_distance: Distance below which points are discarded
            rcs_threshold: Threshold for RCS transformation
        
        Returns:
            Tuple of (enhanced_point_cloud, timestamps). The aggregated point cloud with 
            additional dimensions and timestamps.
        """
        points = np.zeros((cls.nbr_dims(), 0))
        all_pc = cls(points)
        sweep = True

        sample_data_token = sample_rec['data'][chan]
        current_rd_info = nusc.get('sample_data', sample_data_token) 
        i_radar = 0
        
        while sweep:          
            if current_rd_info['is_key_frame'] == True and i_radar != 0: # stop sweeping
                break                  
            
            current_pc = RadarPointCloud.from_file(
                osp.join(nusc.dataroot, current_rd_info['filename']), 
                invalid_states=invalid_states,
                dynprop_states=dynoprop_state,
                ambig_states=ambig_state
            )
            current_pc.remove_close(min_distance)
            
            valid = [p in pdh0 for p in current_pc.points[15, :]]
            current_pc.points = current_pc.points[:, valid]
                                                
            current_pc.remove_close(min_distance)

            if current_pc.points.shape[1] > 0:  # Only add IDs if there are points
                point_ids = np.arange(0, current_pc.points.shape[1], dtype=np.float32).reshape(1, -1)
                radar_id = cls.RADARS_ID.get(chan, 0)  # Default to 0 if channel not found
                radar_ids = np.full((1, current_pc.points.shape[1]), radar_id, dtype=np.float32)
                current_pc.points = np.vstack([current_pc.points, point_ids, radar_ids])
            else:
                if current_rd_info['next'] == '':
                    break
                else:
                    current_rd_info = nusc.get('sample_data', current_rd_info['next'])
                    i_radar += 1
                continue

            current_rd_pose = nusc.get('ego_pose', current_rd_info['ego_pose_token'])
            global_from_car = transform_matrix(
                current_rd_pose['translation'],
                Quaternion(current_rd_pose['rotation']), 
                inverse=False
            )
            
            current_rd_cs = nusc.get('calibrated_sensor', current_rd_info['calibrated_sensor_token'])
            car_from_current = transform_matrix(
                current_rd_cs['translation'], 
                Quaternion(current_rd_cs['rotation']),
                inverse=False
            )
            
            trans_matrix = reduce(np.dot, [global_from_car, car_from_current])
            current_pc.transform(trans_matrix)

            distances_to_ego = cls.compute_distance_to_ego(current_pc.points, current_rd_pose['translation'])
            current_pc.points = np.vstack([current_pc.points, distances_to_ego])
   
            rcs_values = current_pc.points[5, :]  # RCS values
            rcs_areas = cls.compute_rcs_area_for_points(rcs_values, rcs_threshold)
            current_pc.points = np.vstack([current_pc.points, rcs_areas])

            label_row = np.zeros((1, current_pc.points.shape[1]))
            current_pc.points = np.vstack([current_pc.points, label_row])
            
            all_pc.points = np.hstack((all_pc.points, current_pc.points))
            
            if current_rd_info['next'] == '':
                break
            else:
                current_rd_info = nusc.get('sample_data', current_rd_info['next'])

            i_radar += 1
            
        return all_pc 