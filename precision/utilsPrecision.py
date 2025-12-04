import pickle
import os
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from typing import Dict, List
import seaborn as sns
import numpy as np
from statistics import stdev
import argparse

METHOD = 'cca'

def load_clustering_data(scene_name, data_dir=f'../clusterTrack/clustering_data_pkl_rcs_all_review_pdh0', verbose=False):
    """
    Load clustering data for a specific scene from pickle file
    
    Args:
        scene_name: Name of the scene (e.g., 'scene-0061')
        data_dir: Directory containing the pickle files
        
    Returns:
        List of clustering data entries for all cameras and frames
    """
    pickle_filename = os.path.join(data_dir, f"{scene_name}_clustering_data.pkl")
    
    if not os.path.exists(pickle_filename):
        print(f"File not found: {pickle_filename}")
        return None
    
    with open(pickle_filename, 'rb') as f:
        data = pickle.load(f)
    if verbose:
        print(f"Loaded {len(data)} data entries from {pickle_filename}")
    return data

def find_consecutive_clusters(data, min_consecutive_frames=1):
    """
    Find clusters that appear in consecutive frames
    
    Args:
        data: List of clustering data entries
        min_consecutive_frames: Minimum number of consecutive frames required
        
    Returns:
        Dictionary with cluster_id as key and list of consecutive frame sequences
    """
    cluster_frames = defaultdict(list)
    
    for entry in data:
        frame_num = entry['frame_num']
        cluster_ids = entry['cluster_ids']
        
        for cluster_id in cluster_ids:
            cluster_frames[cluster_id].append(frame_num)
    
    consecutive_clusters = {}
    
    for cluster_id, frames in cluster_frames.items():
        frames = sorted(set(frames))  
        
        sequences = []
        current_seq = [frames[0]]
        
        for i in range(1, len(frames)):
            if frames[i] == frames[i-1] + 1: 
                current_seq.append(frames[i])
            else:
                if len(current_seq) >= min_consecutive_frames:
                    sequences.append(current_seq)
                current_seq = [frames[i]]
        
        if len(current_seq) >= min_consecutive_frames:
            sequences.append(current_seq)
        
        if sequences:
            consecutive_clusters[cluster_id] = sequences
    
    return consecutive_clusters

def get_cluster_data_by_frame(data, cluster_id, frame_num):
    """
    Get cluster data for a specific cluster in a specific frame
    """
    for entry in data:
        if entry['frame_num'] == frame_num and cluster_id in entry['cluster_ids']:
            cluster_idx = entry['cluster_ids'].index(cluster_id)
            return {
                'rcs_median': entry['cluster_medians_rcs'][cluster_idx],
                'dyn_prop': entry['cluster_most_common_values'][cluster_idx]
            }
    return None

def transform_rcs(rcs_value):
    """
    Transform radar cross-section from dBsm to square meters.
    RCS_m² = 10^(RCS_dBsm / 10)
    """
    return 10 ** (rcs_value / 10)

def calculate_rcs_similarity(rcs_values, assigned_consecutive_frames, sigma=5):
    """
    Calculate similarity of RCS values across consecutive frames
    Returns value between 0 and 1 (1 = perfect similarity, 0 = no similarity)
    """
    rcs_values_cons = [v for v in rcs_values[0:assigned_consecutive_frames] if v is not None]
    if len(rcs_values_cons) < 2:
        similarity = 1.0
    
    else:
        diffs = np.abs(np.diff(rcs_values_cons))
        mean_diff = diffs.mean()
        similarity = np.exp(-(mean_diff)/sigma)
    return similarity

def calculate_dynprop_similarity(dynprop_values, assigned_consecutive_frames):
    """
    Calculate similarity of dynamic property values across consecutive frames
    Returns value between 0 and 1 (1 = all same, 0 = all different)
    """

    dynprop_values = [v for v in dynprop_values[0:assigned_consecutive_frames] if v is not None]
    if len(dynprop_values) < 2:
        similarity = 1.0

    value_counts = Counter(dynprop_values)
    most_common_count = value_counts.most_common(1)[0][1]
    
    similarity = most_common_count / len(dynprop_values)
    return similarity

def calculate_coefficient(consecutive_frames, assigned_consecutive_frames):
    """
    Calculate coefficient based on how many consecutive frames the cluster appears in
    
    Args:
        consecutive_frames: Number of consecutive frames the cluster actually appears in
        assigned_consecutive_frames: Target number of consecutive frames
        
    Returns:
        Coefficient between 0 and 1
    """
    if consecutive_frames >= assigned_consecutive_frames:
        return 1.0
    else:
        ratio = consecutive_frames / assigned_consecutive_frames
        return ratio

def compute_scene_accuracy(scene_name, assigned_consecutive_frames, sigma, data_dir=f'../clusterTrack/clustering_data_pkl_rcs_all_review_pdh0', verbose=False):
    """
    Compute accuracy metric for a scene based on cluster consistency across consecutive frames
    
    Args:
        scene_name: Name of the scene (e.g., 'scene-0061')
        assigned_consecutive_frames: Target number of consecutive frames
        data_dir: Directory containing the pickle files
        
    Returns:
        Dictionary containing accuracy metrics and detailed analysis
    """
    data = load_clustering_data(scene_name, data_dir)
    if data is None:
        return None
    
    min_consecutive_frames = assigned_consecutive_frames
    consecutive_clusters = find_consecutive_clusters(data, min_consecutive_frames=min_consecutive_frames)
    
    if not consecutive_clusters:
        print(f"No clusters found with consecutive appearances in {scene_name}")
        return {'accuracy': 0.0, 'total_clusters': 0, 'analyzed_clusters': 0}
    
    total_weighted_score = 0.0
    total_frames = len(set(entry['frame_num'] for entry in data))
    analyzed_clusters = 0
    cluster_details = []
    if verbose:
        print(f"Analyzing {len(consecutive_clusters)} clusters with consecutive appearances...")
    
    for cluster_id, sequences in consecutive_clusters.items():
        for sequence in sequences:
            rcs_values = []
            dynprop_values = []
            
            for frame_num in sequence:
                cluster_data = get_cluster_data_by_frame(data, cluster_id, frame_num)
                if cluster_data:
                    transform_rcs_value = transform_rcs(cluster_data['rcs_median'])
                    rcs_values.append(transform_rcs_value)
                    dynprop_values.append(cluster_data['dyn_prop'])
            if len(rcs_values) >= 1:
                rcs_similarity = calculate_rcs_similarity(rcs_values, assigned_consecutive_frames, sigma)
                dynprop_similarity = calculate_dynprop_similarity(dynprop_values, assigned_consecutive_frames)

                consecutive_frames = len(sequence)
                coefficient = calculate_coefficient(consecutive_frames, assigned_consecutive_frames)
                
                combined_similarity = (rcs_similarity + dynprop_similarity) / 2
                weighted_score = combined_similarity * coefficient
                total_weighted_score += weighted_score
                analyzed_clusters += 1

                cluster_details.append({
                    'cluster_id': cluster_id,
                    'sequence': sequence,
                    'consecutive_frames': consecutive_frames,
                    'rcs_values': rcs_values,
                    'dynprop_values': dynprop_values,
                    'rcs_similarity': rcs_similarity,
                    'dynprop_similarity': dynprop_similarity,
                    'coefficient': coefficient,
                    'weighted_score': weighted_score
                })
    
    if analyzed_clusters > 0:
        accuracy = total_weighted_score / analyzed_clusters
    else:
        accuracy = 0.0
    
    results = {
        'scene_name': scene_name,
        'accuracy': accuracy,
        'total_clusters': len(consecutive_clusters),
        'analyzed_clusters': analyzed_clusters,
        'total_frames': total_frames,
        'assigned_consecutive_frames': assigned_consecutive_frames,
        'total_weighted_score': total_weighted_score,
        'cluster_details': cluster_details,
        'consecutive_clusters': consecutive_clusters
    }
    if verbose:
        print(f"Scene {scene_name} Accuracy: {accuracy:.4f}")
    
    return results

def plot_accuracy(F_values: List[int], accuracies: List[float], stds: List[float], output: str = "accuracy_vs_Freq_{METHOD}.png") -> None:
    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.5)
    line = sns.lineplot(x=F_values, y=accuracies, marker="o", linewidth=2, color="#2ca02c", label="MATC-APrecision")
    plt.fill_between(F_values, np.array(accuracies) - np.array(stds), np.array(accuracies) + np.array(stds),
                     color="#2ca02c", alpha=0.2, label="±1 std dev")
    line.set_xlabel("Required consecutive frames $(F_{req})$", fontsize=14)
    line.set_ylabel("MATC-Precision (mean across scenes)", fontsize=14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # line.set_title("Tracking accuracy vs. consecutive-frame requirement")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"Saved figure to {output}")

def plot_summary_precision_sigma(sigma_values, consecutive_frames, nusc):
    all_sigma_results = {}

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for sigma_idx, sigma in enumerate(sigma_values):
        print(f"\nProcessing sigma = {sigma}")
        accuracy = 0
        total_accuracy_list = []
        stds = []
        all_results = {}
        
        for cf in consecutive_frames:
            accuracies_list = [] 
            scene_count = 0
            all_results[cf] = []
            for scene in nusc.scene:
                scene_name = scene['name']
                scene_count += 1
                result = compute_scene_accuracy(scene_name, cf, [sigma])  
                all_results[cf].append(result)
                
                accuracy_value = result['accuracy']
                if hasattr(accuracy_value, 'item'):  
                    accuracy_value = accuracy_value.item()
                elif isinstance(accuracy_value, (list, tuple, np.ndarray)):
                    accuracy_value = float(accuracy_value[0]) if len(accuracy_value) > 0 else 0.0
                elif accuracy_value is None:
                    accuracy_value = 0.0
                
                accuracies_list.append(float(accuracy_value))
                total_accuracy = sum(accuracies_list) / scene_count         
                if scene_count == 85:
                    break
            total_accuracy_list.append(total_accuracy)
            stds.append(stdev(accuracies_list) if len(accuracies_list) > 1 else 0.0)  
            print(f"  F_req={cf:2d} → acc={total_accuracy:.4f}, std={stds[-1]:.4f}")
        
        all_sigma_results[sigma] = {
            'accuracies': total_accuracy_list,
            'stds': stds,
            'color': colors[sigma_idx]
        }

    plt.figure(figsize=(10, 6))

    for sigma in sigma_values:
        results = all_sigma_results[sigma]
        plt.errorbar(consecutive_frames, results['accuracies'], yerr=results['stds'],
                    color=results['color'], linewidth=2, marker='o', 
                    label=f'σ = {sigma}', markersize=4, capsize=3, capthick=1)

    plt.xlabel('Required consecutive frames ($F_{req}$)', fontsize=14)
    plt.ylabel('MATC-Precision (mean across scenes)', fontsize=14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.title('Tracking Accuracy vs. Consecutive-Frame Requirement\nfor Different RCS Sigma Values', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='RCS Sigma (σ)', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'./accuracy_vs_Freq_rcs_dyna_multi_sigma_{METHOD}.png', dpi=300)
    plt.show()

    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for sigma in sigma_values:
        results = all_sigma_results[sigma]
        max_acc = max(results['accuracies'])
        min_acc = min(results['accuracies'])
        avg_acc = np.mean(results['accuracies'])
        print(f"Sigma {sigma:2d}: Max={max_acc:.4f}, Min={min_acc:.4f}, Avg={avg_acc:.4f}")
