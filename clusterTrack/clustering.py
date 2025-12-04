"""
Clustering Module

This module contains clustering algorithms for radar point clouds, including
velocity and position-based clustering with dynamic property analysis.
"""

from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
import numpy as np
from collections import Counter
from sklearn.cluster import DBSCAN


def find_similar_velocities_and_positions_custom(v, x, y, rcs, weighted_rcs, normalized_rcs, 
                                               v_threshold=3, pos_threshold=6.0):
    """
    Find groups where both velocity and position are within thresholds,
    then sub-cluster by dynamic properties (index 4).
    
    This function Groups points based on velocity and position similarity
    
    Args:
        v: Velocity magnitudes for each point
        x: X coordinates for each point
        y: Y coordinates for each point
        rcs: RCS values for each point
        weighted_rcs: Weighted RCS values for each point
        normalized_rcs: Normalized RCS values for each point
        v_threshold: Velocity similarity threshold (default: 3)
        pos_threshold: Position similarity threshold (default: 6.0)
    
    Returns:
        tuple: A tuple containing:
            - similar_groups: List of arrays containing point indices for each cluster
            - cluster_medians_rcs: List of median RCS values for each cluster
            - cluster_median_weighted_rcs: List of median weighted RCS values for each cluster
            - cluster_medians_normalized_rcs: List of median normalized RCS values for each cluster
    """

    # Pairwise distances
    v_distances = pdist(v.reshape(-1, 1))
    pos_distances = pdist(np.column_stack([x, y]))
    
    # Convert to square matrices
    v_matrix = squareform(v_distances)
    pos_matrix = squareform(pos_distances)

    # Similarity masks
    similar_velocity = v_matrix < v_threshold
    similar_position = pos_matrix < pos_threshold
    similar_both = similar_velocity & similar_position

    # Connected components (find clusters)
    n_components, labels = connected_components(csr_matrix(similar_both), directed=False)

    # First stage: velocity and position clustering
    similar_groups = []
    for i in range(n_components):
        component_indices = np.where(labels == i)[0]
        if len(component_indices) > 3:  # Only keep groups with multiple points
            similar_groups.append(component_indices)

    # Cluster statistics
    cluster_medians_rcs = []
    for cluster in similar_groups:
        median_value = np.median(rcs[cluster])
        cluster_medians_rcs.append(median_value)
    
    cluster_median_weighted_rcs = []
    for cluster in similar_groups:
        weighted_rcs_value = np.median(weighted_rcs[cluster])
        cluster_median_weighted_rcs.append(weighted_rcs_value)
    
    cluster_medians_normalized_rcs = []
    for cluster in similar_groups:
        median_value = np.median(normalized_rcs[cluster])
        cluster_medians_normalized_rcs.append(median_value)
    
    return similar_groups, cluster_medians_rcs, cluster_median_weighted_rcs, cluster_medians_normalized_rcs


def dbscan_velocity_position(
    v: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    rcs: np.ndarray,
    weighted_rcs: np.ndarray,
    normalized_rcs: np.ndarray,
    eps_pos: float = 2.0,
    eps_v: float = 0.5,
    min_samples: int = 3,
):
    """
    Cluster points using DBSCAN over a scaled (x, y, v) space.
    
    We scale features so that DBSCAN with eps=1.0 corresponds to the given thresholds:
      X' = [x/eps_pos, y/eps_pos, v/eps_v]
    Then we run DBSCAN(eps=1.0, min_samples=min_samples).
    
    Returns:
        - clusters: list of point index arrays for each DBSCAN cluster (label >= 0)
        - cluster_medians_rcs
        - cluster_median_weighted_rcs
        - cluster_medians_normalized_rcs
    """
    if v.size == 0:
        return [], [], [], []

    X_scaled = np.column_stack([x / float(eps_pos), y / float(eps_pos), v / float(eps_v)])

    db = DBSCAN(eps=1.0, min_samples=int(min_samples))
    labels = db.fit_predict(X_scaled)

    clusters: List[np.ndarray] = []
    unique_labels = sorted([int(l) for l in np.unique(labels) if int(l) >= 0])
    for lbl in unique_labels:
        idx = np.where(labels == lbl)[0]
        if idx.size >= int(min_samples):
            clusters.append(idx)

    cluster_medians_rcs: List[float] = []
    for cluster in clusters:
        cluster_medians_rcs.append(float(np.median(rcs[cluster])))

    cluster_median_weighted_rcs: List[float] = []
    for cluster in clusters:
        cluster_median_weighted_rcs.append(float(np.median(weighted_rcs[cluster])))

    cluster_medians_normalized_rcs: List[float] = []
    for cluster in clusters:
        cluster_medians_normalized_rcs.append(float(np.median(normalized_rcs[cluster])))

    return clusters, cluster_medians_rcs, cluster_median_weighted_rcs, cluster_medians_normalized_rcs


def compute_cluster_features(cluster_indices, radar_points):
    """
    Compute features for a given cluster.
    
    Args:
        cluster_indices: Indices of points in the cluster
        radar_points: Full radar point cloud data
    
    Returns:
        dict: Dictionary containing cluster features
    """
    cluster_points = radar_points[:, cluster_indices]
    features = {}
    
    features['centroid'] = {
        'x': np.mean(cluster_points[0, :]),
        'y': np.mean(cluster_points[1, :]),
        'z': np.mean(cluster_points[2, :])
    }
    
    features['spatial_extent'] = {
        'x_std': np.std(cluster_points[0, :]),
        'y_std': np.std(cluster_points[1, :]),
        'z_std': np.std(cluster_points[2, :])
    }
    
    if radar_points.shape[0] > 8:
        vx, vy = cluster_points[7, :], cluster_points[8, :]
        v_magnitude = np.sqrt(vx**2 + vy**2)
        features['velocity'] = {
            'mean_velocity': np.mean(v_magnitude),
            'velocity_std': np.std(v_magnitude),
            'mean_vx': np.mean(vx),
            'mean_vy': np.mean(vy)
        }
    
    if radar_points.shape[0] > 5:
        rcs = cluster_points[5, :]
        features['rcs'] = {
            'mean_rcs': np.mean(rcs),
            'median_rcs': np.median(rcs),
            'rcs_std': np.std(rcs)
        }
    
    if radar_points.shape[0] > 3:
        dyn_props = cluster_points[3, :]
        counter = Counter(dyn_props)
        most_common_prop, frequency = counter.most_common(1)[0]
        features['dynamic_properties'] = {
            'most_common_property': most_common_prop,
            'property_frequency': frequency,
            'total_points': len(cluster_indices),
            'property_distribution': dict(counter)
        }
    
    features['num_points'] = len(cluster_indices)
    
    return features


def merge_nearby_clusters(clusters, x, y, merge_threshold=5.0):
    """
    Merge clusters that are spatially close to each other.
    
    Args:
        clusters: List of cluster point indices
        x: X coordinates of all points
        y: Y coordinates of all points
        merge_threshold: Distance threshold for merging clusters
    
    Returns:
        List of merged clusters
    """
    if len(clusters) <= 1:
        return clusters
    
    centroids = []
    for cluster in clusters:
        centroid_x = np.mean(x[cluster])
        centroid_y = np.mean(y[cluster])
        centroids.append([centroid_x, centroid_y])
    
    centroids = np.array(centroids)
    
    distances = pdist(centroids)
    distance_matrix = squareform(distances)
    
    merge_mask = distance_matrix < merge_threshold
    n_components, labels = connected_components(csr_matrix(merge_mask), directed=False)
    
    merged_clusters = []
    for i in range(n_components):
        cluster_indices_to_merge = np.where(labels == i)[0]
        merged_cluster = np.concatenate([clusters[j] for j in cluster_indices_to_merge])
        merged_clusters.append(merged_cluster)
    
    return merged_clusters


def filter_clusters_by_size(clusters, min_size=3, max_size=None):
    """
    Filter clusters based on their size.
    
    Args:
        clusters: List of cluster point indices
        min_size: Minimum cluster size (default: 3)
        max_size: Maximum cluster size (optional)
    
    Returns:
        List of filtered clusters
    """
    filtered_clusters = []
    
    for cluster in clusters:
        cluster_size = len(cluster)
        
        if cluster_size < min_size:
            continue
            
        if max_size is not None and cluster_size > max_size:
            continue
            
        filtered_clusters.append(cluster)
    
    return filtered_clusters


def analyze_cluster_stability(clusters_frame1, clusters_frame2, x1, y1, x2, y2, threshold=3.0):
    """
    Analyze stability of clusters between consecutive frames.
    
    Args:
        clusters_frame1: Clusters from frame 1
        clusters_frame2: Clusters from frame 2
        x1, y1: Coordinates from frame 1
        x2, y2: Coordinates from frame 2
        threshold: Distance threshold for matching clusters
    
    Returns:
        dict: Dictionary containing stability analysis results
    """
    centroids1 = []
    for cluster in clusters_frame1:
        if len(cluster) > 0:
            centroid_x = np.mean(x1[cluster])
            centroid_y = np.mean(y1[cluster])
            centroids1.append([centroid_x, centroid_y])
    
    centroids2 = []
    for cluster in clusters_frame2:
        if len(cluster) > 0:
            centroid_x = np.mean(x2[cluster])
            centroid_y = np.mean(y2[cluster])
            centroids2.append([centroid_x, centroid_y])
    
    if len(centroids1) == 0 or len(centroids2) == 0:
        return {
            'matched_clusters': 0,
            'unmatched_frame1': len(centroids1),
            'unmatched_frame2': len(centroids2),
            'stability_ratio': 0.0
        }
    
    centroids1 = np.array(centroids1)
    centroids2 = np.array(centroids2)
    
    matched_clusters = 0
    unmatched_frame1 = len(centroids1)
    unmatched_frame2 = len(centroids2)
    
    for c1 in centroids1:
        distances = np.sqrt(np.sum((centroids2 - c1)**2, axis=1))
        if np.min(distances) < threshold:
            matched_clusters += 1
            unmatched_frame1 -= 1
            min_idx = np.argmin(distances)
            centroids2 = np.delete(centroids2, min_idx, axis=0)
            unmatched_frame2 -= 1
    
    total_clusters = len(clusters_frame1) + len(clusters_frame2)
    stability_ratio = (2 * matched_clusters) / total_clusters if total_clusters > 0 else 0.0
    
    return {
        'matched_clusters': matched_clusters,
        'unmatched_frame1': unmatched_frame1,
        'unmatched_frame2': unmatched_frame2,
        'stability_ratio': stability_ratio
    } 