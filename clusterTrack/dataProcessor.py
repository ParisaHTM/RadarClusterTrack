"""
Data Processor Module

This module contains the main data processing pipeline for radar point cloud analysis
with NuScenes dataset, including scene processing, clustering, and tracking functionality.
"""

import os
import pickle
import numpy as np
from nuscenes.nuscenes import NuScenes
from typing import Dict, List

from clusterTracker import ClusterTracker
from radarPointCloud import RadarPointCloudWithRcsDistance
from radarUtils import get_radar_points_within_lidar_radius, extract_radar_features
from clustering import find_similar_velocities_and_positions_custom, dbscan_velocity_position
from config import (
    CAMS, RADARS_FOR_CAMERA, 
    RADAR_PROCESSING_PARAMS, CLUSTERING_PARAMS, DBSCAN_PARAMS,
    DEFAULT_PATHS, NUSCENES_CONFIG, PROCESSING_FLAGS, TRACKING_PARAMS
)


class RadarDataProcessor:
    """
    Main class for processing radar data from NuScenes dataset.
    
    This class handles the complete pipeline from loading NuScenes data
    to processing radar point clouds, clustering, tracking, and saving results.
    """
    
    def __init__(self, data_path: str = None, config: Dict = None, verbose_class=False):
        """
        Initialize the RadarDataProcessor.
        
        Args:
            data_path: Path to NuScenes dataset
            config: Configuration dictionary (optional)
        """
        self.data_path = data_path
        self.config = config or {}
        
        self.nusc = NuScenes(
            version=NUSCENES_CONFIG['version'], 
            dataroot=self.data_path, 
            verbose=NUSCENES_CONFIG['verbose']
        )
        self.verbose = verbose_class
        
        self.setup_directories()
        
        self.global_cluster_tracker = ClusterTracker(
            max_distance_threshold=TRACKING_PARAMS['max_distance_threshold'],
            cross_camera_threshold=TRACKING_PARAMS['cross_camera_threshold'],
            max_cluster_age=TRACKING_PARAMS['max_cluster_age'],
            max_colors=TRACKING_PARAMS['max_colors']
        )
        
        self.info = {}
        
    def setup_directories(self):
        """Create necessary output directories."""
        self.pickle_save_root = self.config.get('pickle_save_root', DEFAULT_PATHS['pickle_save_root'])
        os.makedirs(self.pickle_save_root, exist_ok=True)
        
    def process_single_scene(self, scene) -> List[Dict]:
        """
        Process a single scene and return clustering data.
        
        Args:
            scene_name: Name of the scene to process
            
        Returns:
            List of dictionaries containing clustering data for each frame/camera
        """
        scene_name = scene['name']
        print(f"Processing scene: {scene_name}")
        
        
        self.info[scene_name] = {}
        scene_clustering_data = []
        
        self.global_cluster_tracker.reset_scene()
        
        sample_rec = self.nusc.get('sample', scene['first_sample_token'])
        sample_in_scene = True
        frame_num = 0
        
        while sample_in_scene:
            if self.verbose:
                print(f"Processing frame {frame_num}")
            
            self.global_cluster_tracker.start_new_frame()
            
            frame_data = self.process_single_frame(sample_rec, scene_name, frame_num)
            scene_clustering_data.extend(frame_data)
            
            self.global_cluster_tracker.finalize_frame()
            
            if sample_rec['next'] == '':
                sample_in_scene = False
            else:
                sample_rec = self.nusc.get('sample', sample_rec['next'])
                frame_num += 1
        
        print(f"=== Scene {scene_name} completed ===")
       
        return scene_clustering_data
    
    def process_single_frame(self, sample_rec: Dict, scene_name: str, frame_num: int) -> List[Dict]:
        """
        Process a single frame across all cameras.
        
        Args:
            sample_rec: NuScenes sample record
            scene_name: Name of the current scene
            frame_num: Frame number
            
        Returns:
            List of camera clustering data for this frame
        """
        frame_data = []
        sample_token = sample_rec['token']
        
        lidar_record = sample_rec['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_record)
        lidar_pose_record = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        ego_x = lidar_pose_record['translation'][0]
        ego_y = lidar_pose_record['translation'][1]
        
        for cam in CAMS:
            camera_data = self.process_camera_view(
                sample_rec, cam, scene_name, frame_num, sample_token, ego_x, ego_y
            )
            frame_data.append(camera_data)
            
        return frame_data
    
    def process_camera_view(self, sample_rec: Dict, cam: str, scene_name: str, 
                          frame_num: int, sample_token: str, ego_x: float, ego_y: float) -> Dict:
        """
        Process radar data for a specific camera view.
        
        Args:
            sample_rec: NuScenes sample record
            cam: Camera name
            scene_name: Name of the current scene
            frame_num: Frame number
            sample_token: Sample token
            ego_x, ego_y: Ego vehicle position
            
        Returns:
            Dictionary containing processing results for this camera view
        """
        
        all_radar_pcs = RadarPointCloudWithRcsDistance(np.zeros((23, 0)))
        
        for radar_chan in RADARS_FOR_CAMERA[cam]:
            all_pc = RadarPointCloudWithRcsDistance.from_file_multisweep(
                self.nusc, sample_rec, radar_chan,
                **RADAR_PROCESSING_PARAMS
            )
            all_radar_pcs.points = np.hstack((all_radar_pcs.points, all_pc.points))
        
        filtered_radar_points, mask, lidar_global_pos, distances = get_radar_points_within_lidar_radius(
            all_radar_pcs, self.nusc, sample_rec, radius=50.0
        )
        
        if filtered_radar_points.shape[1] == 0:
            return self.create_empty_camera_data(
                sample_token, scene_name, frame_num, cam, ego_x, ego_y
            )
        
        features = extract_radar_features(filtered_radar_points)
        
        method = CLUSTERING_PARAMS.get('method', 'cca')
        if method == 'cca':
            clusters, cluster_medians_rcs, cluster_median_weighted_rcs, cluster_medians_normalized_rcs = \
                find_similar_velocities_and_positions_custom(
                    features['v'], features['x'], features['y'], 
                    features['rcs'], features.get('weighted_rcs', features['rcs']), 
                    features['normalized_rcs'],
                    v_threshold=CLUSTERING_PARAMS['velocity_threshold'],
                    pos_threshold=CLUSTERING_PARAMS['position_threshold']
                )
        elif method == 'dbscan':
            clusters, cluster_medians_rcs, cluster_median_weighted_rcs, cluster_medians_normalized_rcs = \
                dbscan_velocity_position(
                    features['v'], features['x'], features['y'], 
                    features['rcs'], features.get('weighted_rcs', features['rcs']), 
                    features['normalized_rcs'],
                    eps_pos=float(DBSCAN_PARAMS.get('eps_pos', 2.0)),
                    eps_v=float(DBSCAN_PARAMS.get('eps_v', 0.5)),
                    min_samples=int(DBSCAN_PARAMS.get('min_samples', 3)),
                )

        clusters, cluster_medians_rcs, cluster_median_weighted_rcs, cluster_medians_normalized_rcs = \
            find_similar_velocities_and_positions_custom(
                features['v'], features['x'], features['y'], 
                features['rcs'], features.get('weighted_rcs', features['rcs']), 
                features['normalized_rcs'],
                v_threshold=CLUSTERING_PARAMS['velocity_threshold'],
                pos_threshold=CLUSTERING_PARAMS['position_threshold']
            )
        
        cluster_unique_ids, cluster_color_indices, cluster_most_common_values, \
        cluster_dyn_prop_labels, rcs_clusters_medians, all_rcs_values, \
        all_filtered_radar_points = self.global_cluster_tracker.track_clusters(cam,       
                                                                                clusters, 
                                                                                features['rcs'],
                                                                                filtered_radar_points,
                                                                                features['x'], features['y'], 
                                                                                features['dyn_prop'])

        camera_clustering_data = {
            'sample_token': sample_token,
            'scene_name': scene_name,
            'frame_num': frame_num,
            'camera': cam,
            'clusters': clusters,
            'cluster_ids': cluster_unique_ids,
            'cluster_color_indices': cluster_color_indices,
            'radar_points': all_filtered_radar_points,
            'cluster_global_positions': {
                'x': features['x'],
                'y': features['y']
            },
            'cluster_medians_rcs': rcs_clusters_medians,
            'ego_position': {
                'x': ego_x,
                'y': ego_y
            },
            'cluster_velocities': features['v'],
            'rcs_values': all_rcs_values,
            'cluster_dyn_prop_labels': cluster_dyn_prop_labels,
            'cluster_most_common_values': cluster_most_common_values,
            'num_radar_points': filtered_radar_points.shape[1],
            'num_clusters': len(clusters)
        }
        
        if self.verbose:
            print(f"Frame {frame_num}, Camera {cam}: {len(clusters)} clusters, "
              f"{filtered_radar_points.shape[1]} radar points")
        
        return camera_clustering_data
    
    def create_empty_camera_data(self, sample_token: str, scene_name: str, 
                                frame_num: int, cam: str, ego_x: float, ego_y: float) -> Dict:
        """
        Create empty camera data structure when no radar points are available.
        
        Args:
            sample_token: Sample token
            scene_name: Scene name
            frame_num: Frame number
            cam: Camera name
            ego_x, ego_y: Ego position
            
        Returns:
            Empty camera data dictionary
        """
        return {
            'sample_token': sample_token,
            'scene_name': scene_name,
            'frame_num': frame_num,
            'camera': cam,
            'clusters': [],
            'cluster_ids': [],
            'cluster_color_indices': [],
            'radar_points': np.zeros((23, 0)),
            'cluster_global_positions': {'x': np.array([]), 'y': np.array([])},
            'cluster_medians_rcs': [],
            'ego_position': {'x': ego_x, 'y': ego_y},
            'cluster_velocities': np.array([]),
            'rcs_values': np.array([]),
            'cluster_dyn_prop_labels': [],
            'cluster_most_common_values': [],
            'num_radar_points': 0,
            'num_clusters': 0
        }
    
    def save_clustering_data(self, scene_clustering_data: List[Dict], scene_name: str):
        """
        Save clustering data to pickle file.
        
        Args:
            scene_clustering_data: List of clustering data dictionaries
            scene_name: Name of the scene
        """
        if PROCESSING_FLAGS['save_pickle']:
            pickle_filename = os.path.join(self.pickle_save_root, f"{scene_name}_clustering_data.pkl")
            with open(pickle_filename, 'wb') as f:
                pickle.dump(scene_clustering_data, f)
            print(f"Saved clustering data to: {pickle_filename}")
            print(f"Total data entries saved: {len(scene_clustering_data)}")
    
    def process_scenes(self, scene_names: List[str] = None, target_scene: str = None):
        """
        Process multiple scenes or a single target scene.
        
        Args:
            scene_names: List of scene names to process (optional)
            target_scene: Single scene name to process (optional)
            
        Note: If both scene_names and target_scene are None, all scenes will be processed.
        """

        all_scenes_names = self.get_available_scenes().keys
        all_scenes = self.get_available_scenes()

        if target_scene:
            scene = None
            if target_scene in all_scenes_names:
                scene == target_scene           
            if scene is None:
                raise ValueError(f"Scene {scene_name} not found") 
            scene_names = [target_scene]
        elif scene_names:
            scene_names = scene_names
        elif scene_names is None:
            scene_names = all_scenes
            print(f"No specific scenes provided. Processing all {len(scene_names)} scenes...")
        
        for i, scene_name in enumerate(scene_names, 1):
            try:
                print(f"Processing scene {i}/{len(scene_names)}: {scene_name}")
                scene_clustering_data = self.process_single_scene(all_scenes[scene_name])
                self.save_clustering_data(scene_clustering_data, scene_name)
                print(f"Completed scene {i}/{len(scene_names)}: {scene_name}")
            except Exception as e:
                print(f"Error processing scene {scene_name}: {str(e)}")
                continue

    def get_available_scenes(self):
        """
        Get list of all available scene names in the dataset.
        
        Returns:
            List of scene names
        """
        return {scene['name']: scene for scene in self.nusc.scene}
