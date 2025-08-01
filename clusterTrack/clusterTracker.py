import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter

class ClusterTracker:
    """
    Cluster tracker
    dDstance-based matching and temporal persistence.
    """
    
    def __init__(self, 
                 max_distance_threshold: float = 5.0,
                 cross_camera_threshold: float = 3.0,
                 max_cluster_age: int = 10,
                 max_colors: int = 50,
                 verbose: bool = False):
        """
        Initialize cluster tracker
        
        Args:
            max_distance_threshold: Max distance for temporal matching (meters)
            cross_camera_threshold: Max distance for cross-camera matching (meters)
            max_cluster_age: Max frames to keep a cluster alive without detection
            max_colors: Maximum number of colors to cycle through
        """
        self.max_distance_threshold = max_distance_threshold
        self.cross_camera_threshold = cross_camera_threshold
        self.max_cluster_age = max_cluster_age
        self.max_colors = max_colors
        
        # Cluster tracking
        self.active_clusters = {}  # cluster_id -> ClusterInfo
        self.frame_count = 0
        self.scene_cluster_count = 0
        
        # Current frame tracking
        self.current_frame_clusters = {}  # camera_name -> [(cluster_id, position)]
        
        self.cluster_colors = {}
        self.available_colors = list(range(max_colors))

        self.verbose = verbose
        
    def reset_scene(self):
        """Reset tracker for new scene"""
        self.active_clusters.clear()
        self.current_frame_clusters.clear()
        self.cluster_colors.clear()
        self.available_colors = list(range(self.max_colors))
        self.frame_count = 0
        self.scene_cluster_count = 0
        print("Cluster tracker reset for new scene")
        
    def start_new_frame(self):
        """Start processing a new frame"""
        self.frame_count += 1
        self.current_frame_clusters.clear()
        
        # Age existing clusters and remove old ones
        clusters_to_remove = []
        for cluster_id, cluster_info in self.active_clusters.items():
            cluster_info.age += 1
            if cluster_info.age > self.max_cluster_age:
                clusters_to_remove.append(cluster_id)
                
        # Remove aged out clusters
        for cluster_id in clusters_to_remove:
            self._release_cluster(cluster_id)
            
    def _release_cluster(self, cluster_id: int):
        """Release a cluster and its color"""
        if cluster_id in self.active_clusters:
            del self.active_clusters[cluster_id]
            
        if cluster_id in self.cluster_colors:
            color = self.cluster_colors.pop(cluster_id)
            if color not in self.available_colors:
                self.available_colors.append(color)
                
    def _get_new_cluster_id(self) -> int:
        """Get a new unique cluster ID"""
        new_id = self.scene_cluster_count
        self.scene_cluster_count += 1
        return new_id
        
    def _assign_color(self, cluster_id: int) -> int:
        """Assign color to cluster"""
        if cluster_id in self.cluster_colors:
            return self.cluster_colors[cluster_id]
            
        if self.available_colors:
            color = self.available_colors.pop(0)
        else:
            color = cluster_id % self.max_colors
            
        self.cluster_colors[cluster_id] = color
        return color
        
    def _calculate_cluster_centroid(self, cluster_points_x, cluster_points_y) -> np.ndarray:
        """Calculate cluster centroid"""
        return np.array([np.mean(cluster_points_x), np.mean(cluster_points_y)])
        
    def _find_best_match(self, current_position: np.ndarray, camera_name: str) -> Tuple[Optional[int], float]:
        """
        Find the best matching cluster using ONLY distance-based matching (no velocity)
        
        Returns:
            (cluster_id, distance) or (None, inf) if no match found
        """
        best_match_id = None
        best_distance = float('inf')
        
        # First check current frame (cross-camera matching)
        for cam_name, cam_clusters in self.current_frame_clusters.items():
            if cam_name != camera_name:
                for cluster_id, position in cam_clusters:
                    distance = np.linalg.norm(current_position - position)
                    if distance <= self.cross_camera_threshold and distance < best_distance:
                        best_distance = distance
                        best_match_id = cluster_id
                        
        # If no cross-camera match, check temporal matches (Distance only)
        if best_match_id is None:
            for cluster_id, cluster_info in self.active_clusters.items():
                # Distance-based matching
                distance = np.linalg.norm(current_position - cluster_info.position)
                
                if distance <= self.max_distance_threshold and distance < best_distance:
                    best_distance = distance
                    best_match_id = cluster_id
                    
        return best_match_id, best_distance
        
    def track_clusters(self, camera_name: str, clusters: List, 
                      rcs: np.ndarray,
                      filtered_radar_points: np.ndarray,
                      cluster_points_x: np.ndarray, cluster_points_y: np.ndarray,
                      cluster_dyn_prop: np.ndarray) -> Tuple[List[int], List[int], List[int], List[int]]:
        """
        Track clusters using distance-based matching
        """
        cluster_ids = []
        color_indices = []
        most_common_values = []
        dyn_prop_labels = []
        rcs_clusters_medians = []
        all_rcs_values = []
        all_filtered_radar_points = []
        
        if camera_name not in self.current_frame_clusters:
            self.current_frame_clusters[camera_name] = []

        if self.verbose:
            print(f"Frame {self.frame_count}, Camera {camera_name}: Processing {len(clusters)} clusters")
        
        for i, cluster in enumerate(clusters):
            cluster_x = cluster_points_x[cluster]
            cluster_y = cluster_points_y[cluster]
            cluster_dyn_prop_cluster = cluster_dyn_prop[cluster]
            cluster_filtered_radar_points = filtered_radar_points[:, cluster]
            all_filtered_radar_points.append(cluster_filtered_radar_points)
            counter = Counter(cluster_dyn_prop_cluster)
            rcs_cluster = rcs[cluster]
            rcs_cluster_median = np.median(rcs_cluster)
            all_rcs_values.append(rcs_cluster)
            most_common_value, frequency = counter.most_common(1)[0]
            current_position = self._calculate_cluster_centroid(cluster_x, cluster_y)
            
            matched_id, distance = self._find_best_match(current_position, camera_name)
            
            if matched_id is not None:
                cluster_id = matched_id
                cluster_info = self.active_clusters[cluster_id]

                cluster_info.position = current_position
                cluster_info.age = 0
                cluster_info.last_seen_frame = self.frame_count
                cluster_info.cameras_seen.add(camera_name)
                
                if self.verbose:
                    print(f"  Cluster {i}: Matched to existing ID {cluster_id} (distance: {distance:.2f}m)")
                
            else:
                cluster_id = self._get_new_cluster_id()
                cluster_info = ClusterInfo(
                    position=current_position,
                    age=0,
                    last_seen_frame=self.frame_count,
                    cameras_seen={camera_name}
                )
                self.active_clusters[cluster_id] = cluster_info
                if self.verbose:
                    print(f"  Cluster {i}: New cluster ID {cluster_id} created")

            self.current_frame_clusters[camera_name].append((cluster_id, current_position))
            color_index = self._assign_color(cluster_id)
            
            cluster_ids.append(cluster_id)
            color_indices.append(color_index)
            most_common_values.append(most_common_value)
            dyn_prop_labels.append(most_common_value)
            rcs_clusters_medians.append(rcs_cluster_median)

            if self.verbose:
                print(f"  Active clusters: {len(self.active_clusters)}")
                print(f"  Total scene clusters created: {self.scene_cluster_count}")
        
        return cluster_ids, color_indices, most_common_values, dyn_prop_labels, rcs_clusters_medians, all_rcs_values, all_filtered_radar_points
        
    def finalize_frame(self):
        """Finalize frame processing"""
        # Age clusters that weren't seen this frame
        seen_cluster_ids = set()
        for cam_clusters in self.current_frame_clusters.values():
            for cluster_id, _ in cam_clusters:
                seen_cluster_ids.add(cluster_id)
                
        for cluster_id, cluster_info in self.active_clusters.items():
            if cluster_id not in seen_cluster_ids:
                cluster_info.age += 1
                
    def get_stats(self) -> Dict:
        """Get tracking statistics"""
        return {
            'frame': self.frame_count,
            'active_clusters': len(self.active_clusters),
            'total_scene_clusters': self.scene_cluster_count,
            'clusters_by_age': {age: sum(1 for c in self.active_clusters.values() if c.age == age) 
                               for age in range(self.max_cluster_age + 1)}
        }


class ClusterInfo:
    """Cluster info without velocity"""
    def __init__(self, position: np.ndarray, age: int = 0, 
                 last_seen_frame: int = 0, cameras_seen: set = None):
        self.position = position
        self.age = age
        self.last_seen_frame = last_seen_frame
        self.cameras_seen = cameras_seen or set()
