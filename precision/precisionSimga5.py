from utilsPrecision import compute_scene_accuracy, plot_accuracy
from statistics import stdev
import matplotlib.pyplot as plt
from collections import Counter
from collections import defaultdict
import numpy as np

def precisionSimga5(nusc, consecutive_frames = range(1, 30), sigma=5):
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
            result = compute_scene_accuracy(scene_name, cf, sigma)
            all_results[cf].append(result)
            accuracies_list.append(result['accuracy'])       
            total_accuracy = sum(accuracies_list) / scene_count         
            if scene_count == 85:
                break
        total_accuracy_list.append(total_accuracy)
        stds.append(stdev(accuracies_list) if len(accuracies_list) > 1 else 0.0)  
        print(f"F_req={cf:2d} → acc={total_accuracy:.4f}, std={stds[-1]:.4f}")


    plot_accuracy(consecutive_frames, total_accuracy_list, stds, output='./accuracy_vs_Freq_rcs_dyna.png')
    print("The plot is saved in the current directory")
    return all_results

def plot_num_cluster(all_results):
    # Set style for better visuals
    plt.style.use('default')
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    # Professional color palette (same as sigma plot)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    # Initialize list to store total clusters for each cf
    total_clusters_list = []

    # Loop through each cf index
    for cf in range(len(all_results)):
        total_cluster_count = 0
        for scene in all_results[cf + 1]:
            total_cluster_count += scene['analyzed_clusters']
        total_clusters_list.append(total_cluster_count)

    cf_indices = list(range(1, len(all_results) + 1))

    plt.figure(figsize=(24, 12))
    bars = plt.bar(cf_indices, total_clusters_list, color=colors[0], alpha=0.8, edgecolor='white', linewidth=0.5)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height)}', ha='center', va='bottom', fontsize=22, color='black')

    plt.xlabel('Consecutive Frame Requirements ($F_{req}$)', fontsize=22, fontweight='bold')
    plt.ylabel('Total Number of Clusters', fontsize=22, fontweight='bold')
    # plt.title('Total Analyzed Clusters per Consecutive Frame Requirement', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(cf_indices, fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('num_cluster.png')
    print("The plot is saved in the current directory")
    plt.show()
    return total_clusters_list, cf_indices

def plot_mov_stationary_cluster(all_results):

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    moving_precision = defaultdict(int)
    stationary_precision = defaultdict(int)
    oncoming_precision = defaultdict(int)
    stationary_candidate_precision = defaultdict(int)
    unknown_precision = defaultdict(int)
    cross_stationary_precision = defaultdict(int)
    cross_moving_precision = defaultdict(int)
    stopped_precision = defaultdict(int)

    moving_percision_score = defaultdict(list)
    stationary_percision_score = defaultdict(list)
    oncoming_percision_score = defaultdict(list)
    stationary_candidate_percision_score = defaultdict(list)
    unknown_percision_score = defaultdict(list)
    cross_stationary_percision_score = defaultdict(list)
    cross_moving_percision_score = defaultdict(list)

    # Collect data
    for cf in range(len(all_results)):
        cf_idx = cf + 1
        
        for scene in all_results[cf_idx]:
            if 'cluster_details' in scene:
                for cluster_detail in scene['cluster_details']:
                    counter = Counter(cluster_detail['dynprop_values'])
                    most_common_value, _ = counter.most_common(1)[0]

                    if most_common_value == 0:
                        moving_precision[cf_idx] += 1
                        moving_percision_score[cf_idx].append(cluster_detail['weighted_score'])
                    elif most_common_value == 1:
                        stationary_precision[cf_idx] += 1
                        stationary_percision_score[cf_idx].append(cluster_detail['weighted_score'])
                    elif most_common_value == 2:
                        oncoming_precision[cf_idx] += 1
                        oncoming_percision_score[cf_idx].append(cluster_detail['weighted_score'])
                    elif most_common_value == 3:
                        stationary_candidate_precision[cf_idx] += 1
                        stationary_candidate_percision_score[cf_idx].append(cluster_detail['weighted_score'])
                    elif most_common_value == 4:
                        unknown_precision[cf_idx] += 1
                        unknown_percision_score[cf_idx].append(cluster_detail['weighted_score'])
                    elif most_common_value == 5:
                        cross_stationary_precision[cf_idx] += 1
                        cross_stationary_percision_score[cf_idx].append(cluster_detail['weighted_score'])
                    elif most_common_value == 6:
                        cross_moving_precision[cf_idx] += 1
                        cross_moving_percision_score[cf_idx].append(cluster_detail['weighted_score'])
                    elif most_common_value == 7:
                        stopped_precision[cf_idx] += 1

    # Motion categories with improved colors
    motion_categories = [
        ("Moving", moving_precision, colors[0]),
        ("Stationary", stationary_precision, colors[1]),
        ("Oncoming", oncoming_precision, colors[2]),
        ("Stationary Candidate", stationary_candidate_precision, colors[3]),
        ("Unknown", unknown_precision, colors[4]),
        ("Cross Stationary", cross_stationary_precision, colors[5]),
        ("Cross Moving", cross_moving_precision, colors[6]),
        ("Stopped", stopped_precision, colors[7])
    ]

    # Create improved stacked bar plot
    cf_indices = list(range(1, len(all_results) + 1))
    bottom = [0] * len(cf_indices)
    plt.figure(figsize=(14, 8))

    for label, data_dict, color in motion_categories:
        heights = [data_dict[cf] for cf in cf_indices]
        plt.bar(cf_indices, heights, bottom=bottom, label=label, color=color, alpha=0.8, edgecolor='white', linewidth=0.3)
        bottom = [b + h for b, h in zip(bottom, heights)]

    plt.xlabel('Consecutive Frame Requirements ($F_{req}$)', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Clusters', fontsize=14, fontweight='bold')
    # plt.title('Distribution of Cluster Motion Types by Consecutive Frame Requirements', fontsize=16, fontweight='bold', pad=20)
    plt.xticks([cf for cf in cf_indices if cf % 2 == 0], fontsize=20)
    plt.yticks(fontsize=20)
    # plt.legend(title='Motion Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=13)
    plt.legend(title='Motion Type', loc='upper right', fontsize=22, title_fontsize=22, frameon=True)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('mov_stationary_cluster.png')
    print("The plot is saved in the current directory")
    plt.show()
    return motion_categories, cf_indices


def plot_percentage_mov_status_cluster(motion_categories, total_clusters_list, cf_indices):
    percent_data = {}
    for label, data_dict, _ in motion_categories:
        percent_data[label] = [
            (data_dict[cf] / total_clusters_list[cf - 1] * 100 if total_clusters_list[cf - 1] > 0 else 0)
            for cf in cf_indices
        ]

    bottom = [0] * len(cf_indices)
    plt.figure(figsize=(14, 8))

    for label, _, color in motion_categories:
        heights = percent_data[label]
        plt.bar(cf_indices, heights, bottom=bottom, label=label, color=color, alpha=0.8, edgecolor='white', linewidth=0.3)
        bottom = [b + h for b, h in zip(bottom, heights)]

    plt.xlabel('Consecutive Frame Requirements ($F_{req}$)', fontsize=14, fontweight='bold')
    plt.ylabel('Percentage of Clusters (%)', fontsize=14, fontweight='bold')
    # plt.title('Percentage Distribution of Cluster Motion Types by Consecutive Frame Requirements', fontsize=16, fontweight='bold', pad=20)
    plt.xticks([cf for cf in cf_indices if cf % 2 == 0], fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(0, 100)
    # plt.legend(title='Motion Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=13)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_precistion_mov_status_cluster(motion_categories, cf_indices):

    motion_score_categories = [
        ("Moving", motion_categories[0][1], 'steelblue'),
        ("Stationary", motion_categories[1][1], 'orange'),
        ("Oncoming", motion_categories[2][1], 'green'),
        ("Stationary Candidate", motion_categories[3][1], 'red'),
        ("Unknown", motion_categories[4][1], 'purple'),
        ("Cross Stationary", motion_categories[5][1], 'brown'),
        ("Cross Moving", motion_categories[6][1], 'pink')
    ]

    plt.figure(figsize=(10, 6))

    # Plot each category with mean, min, and max lines
    for label, score_dict, color in motion_score_categories:
        means, mins, maxs = [], [], []

        for cf in cf_indices:
            scores = score_dict[cf]
            if scores:
                means.append(np.mean(scores))
                mins.append(np.min(scores))
                maxs.append(np.max(scores))
            else:
                means.append(np.nan)
                mins.append(np.nan)
                maxs.append(np.nan)

        means = np.array(means)
        mins = np.array(mins)
        maxs = np.array(maxs)

        # Plot mean (solid line)
        plt.plot(cf_indices, means, label=f'{label} Mean', color=color, marker='o')

        # Plot min and max (dashed lines)
        plt.plot(cf_indices, mins, linestyle='--', color=color, alpha=0.6, label=f'{label} Min')
        plt.plot(cf_indices, maxs, linestyle='--', color=color, alpha=0.6, label=f'{label} Max')

    plt.xlabel('Consecutive Frames ($F_{req}$)', fontsize=14, fontweight='bold')
    plt.ylabel(r'Weighted Score ($WS_{i}$)', fontsize=14, fontweight='bold')

    # plt.title('Mean, Min, Max of Weighted Scores per Consecutive Frames', fontsize=14)
    plt.xticks([cf for cf in cf_indices if cf % 2 == 0], fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.5)
    # plt.legend(title='Motion Type (Mean / Min / Max)', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=18)
    plt.tight_layout()
    plt.savefig('precistion_mov_status_cluster.png')
    plt.show()
    print("The plot is saved in the current directory")

def plot_rcs_cluster(all_results, cf_indices):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    rcs_0_10_count = defaultdict(int)
    rcs_10_15_count = defaultdict(int)
    rcs_15_20_count = defaultdict(int)
    rcs_more_20_count = defaultdict(int)

    rcs_0_10_score = defaultdict(list)
    rcs_10_15_score = defaultdict(list)
    rcs_15_20_score = defaultdict(list)
    rcs_more_20_score = defaultdict(list)

    for cf in range(len(all_results)):
        cf_idx = cf + 1
        
        for scene in all_results[cf_idx]:
            if 'cluster_details' in scene:
                for cluster_detail in scene['cluster_details']:
                    rcs_median = np.median(cluster_detail['rcs_values'])

                    if rcs_median < 10:
                        rcs_0_10_count[cf_idx] += 1
                        rcs_0_10_score[cf_idx].append(cluster_detail['weighted_score'])
                    elif rcs_median < 15:
                        rcs_10_15_count[cf_idx] += 1
                        rcs_10_15_score[cf_idx].append(cluster_detail['weighted_score'])
                    elif rcs_median < 20:
                        rcs_15_20_count[cf_idx] += 1
                        rcs_15_20_score[cf_idx].append(cluster_detail['weighted_score'])
                    else:
                        rcs_more_20_count[cf_idx] += 1
                        rcs_more_20_score[cf_idx].append(cluster_detail['weighted_score'])

    rcs_categories = [
        ("0 ≤ RCS < 10 dBsm", rcs_0_10_count, colors[0]),
        ("10 ≤ RCS < 15 dBsm", rcs_10_15_count, colors[1]),
        ("15 ≤ RCS < 20 dBsm", rcs_15_20_count, colors[2]),
        ("RCS ≥ 20 dBsm", rcs_more_20_count, colors[3])
    ]

    bottom = [0] * len(cf_indices)
    plt.figure(figsize=(14, 8))

    for label, count_dict, color in rcs_categories:
        counts = [count_dict[cf] for cf in cf_indices]
        plt.bar(cf_indices, counts, bottom=bottom, label=label, color=color, alpha=0.8, edgecolor='white', linewidth=0.3)
        bottom = [b + c for b, c in zip(bottom, counts)]

    plt.xlabel('Consecutive Frame Requirements ($F_{req}$)', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Clusters', fontsize=14, fontweight='bold')
    plt.xticks([cf for cf in cf_indices if cf % 2 == 0], fontsize=20)
    plt.yticks(fontsize=20)
    # plt.legend(title='RCS Range', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=13)
    plt.legend(title='RCS Range', loc='upper right', fontsize=22, title_fontsize=22, frameon=True)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('rcs_cluster.png')
    plt.show()
    print("The plot is saved in the current directory")
    return rcs_categories, cf_indices

def plot_percentage_rcs_status_cluster(rcs_categories, total_clusters_list, cf_indices):
    # Compute percentages for RCS ranges
    rcs_percent_data = {}
    for label, count_dict, _ in rcs_categories:
        rcs_percent_data[label] = [
            (count_dict[cf] / total_clusters_list[cf - 1] * 100 if total_clusters_list[cf - 1] > 0 else 0)
            for cf in cf_indices
        ]

    # Create improved percentage stacked bar chart for RCS
    bottom = [0] * len(cf_indices)
    plt.figure(figsize=(14, 8))

    for label, _, color in rcs_categories:
        heights = rcs_percent_data[label]
        plt.bar(cf_indices, heights, bottom=bottom, label=label, color=color, alpha=0.8, edgecolor='white', linewidth=0.3)
        bottom = [b + h for b, h in zip(bottom, heights)]

    plt.xlabel('Consecutive Frame Requirements ($F_{req}$)', fontsize=14, fontweight='bold')
    plt.ylabel('Percentage of Clusters (%)', fontsize=14, fontweight='bold')
    # plt.title('Percentage Distribution of Clusters by RCS Range and Consecutive Frame Requirements', fontsize=16, fontweight='bold', pad=20)
    plt.xticks([cf for cf in cf_indices if cf % 2 == 0], fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(0, 100)
    # plt.legend(title='RCS Range', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=13)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('percentage_rcs_status_cluster.png')
    plt.show()
    print("The plot is saved in the current directory")

def plot_precision_rcs_status_cluster(rcs_categories, cf_indices):
    # Line plot setup

    rcs_score_categories = [
        ("0 ≤ RCS < 10", rcs_categories[0][1], 'steelblue'),
        ("10 ≤ RCS < 15", rcs_categories[1][1], 'green'),
        ("15 ≤ RCS < 20", rcs_categories[2][1], 'orange'),
        ("RCS ≥ 20", rcs_categories[3][1], 'red')
    ]

    plt.figure(figsize=(10, 6))

    for label, score_dict, color in rcs_score_categories:
        means, mins, maxs = [], [], []
        for cf in cf_indices:
            scores = score_dict[cf]
            if scores:
                means.append(np.mean(scores))
                mins.append(np.min(scores))
                maxs.append(np.max(scores))
            else:
                means.append(np.nan)
                mins.append(np.nan)
                maxs.append(np.nan)

        plt.plot(cf_indices, means, label=f'{label} Mean', color=color, marker='o')
        plt.plot(cf_indices, mins, linestyle='--', color=color, alpha=0.5, label=f'{label} Min')
        plt.plot(cf_indices, maxs, linestyle='--', color=color, alpha=0.5, label=f'{label} Max')

    plt.xlabel('Consecutive Frames ($F_{req}$)', fontsize=14, fontweight='bold')
    plt.ylabel(r'Weighted Score ($\mathcal{WS}_{i}$)', fontsize=14, fontweight='bold')
    # plt.title('Mean, Min, and Max of Weighted Scores by RCS Range per Consecutive Frames', fontsize=14)
    plt.xticks([cf for cf in cf_indices if cf % 2 == 0], fontsize=20)
    plt.yticks(fontsize=20)
    # plt.legend(title='RCS Range (Mean/Min/Max)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('precision_rcs_status_cluster.png')
    plt.show()
    print("The plot is saved in the current directory")