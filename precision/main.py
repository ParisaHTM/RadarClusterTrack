import argparse
from precisionSimga5 import *
from utilsPrecision import *
from nuscenes.nuscenes import NuScenes

def parse_args():
    parser = argparse.ArgumentParser(description="Precision Evaluation")
    parser.add_argument('--sigma', type=int, default=5,
                        help="Choose sigma value for single evaluation")
    parser.add_argument('--data-path', type=str, default='C:/Users/qxy699/Documents/GAFusion/nuscences/nuScenes-lidarseg-all-v1.0',
                        help="Path to the data directory")
    parser.add_argument('--version', type=str, default="v1.0-trainval",
                        help="Choose version of the dataset")
    parser.add_argument('--range-sigma', type=List, default=[1, 3, 5, 7, 10],
                        help="Choose sigma ranges a alist of integers")
    parser.add_argument('--consecutive-frames', type=List, default=range(1, 30),
                        help="Range of consecutive frames to evaluate")
    parser.add_argument('--plot_type', type=str, choices=['num_cluster', 'precisionSimga5', 'mov_stationary_cluster', 
                                                     'percentage_mov_status_cluster', 
                                                     'precistion_mov_status_cluster', 
                                                     'rcs_cluster', 'percentage_rcs_status_cluster', 
                                                     'precision_rcs_status_cluster'],
                        help="Choose from choices; precisionSimga5, num_cluster, mov_stationary_cluster, \
                            percentage_mov_status_cluster, precistion_mov_status_cluster, \
                            rcs_cluster, percentage_rcs_status_cluster, precision_rcs_status_cluster")
    return parser.parse_args()

def main():
    args = parse_args()
    nusc = NuScenes(version=args.version, dataroot=args.data_path, verbose=True)
    all_results = precisionSimga5(nusc, args.consecutive_frames, args.sigma)
    cf_indices = list(range(1, len(all_results) + 1))
    if args.plot_type == 'precisionSimga5':
        all_results = all_results
    elif args.plot_type == 'num_cluster':
        # all_results = precisionSimga5(nusc, args.consecutive_frames, args.sigma)
        total_clusters_list, cf_indices = plot_num_cluster(all_results)
    elif args.plot_type == 'mov_stationary_cluster':
        # all_results = precisionSimga5(nusc, args.consecutive_frames, args.sigma)
        motion_categories, cf_indices = plot_mov_stationary_cluster(all_results)
    elif args.plot_type == 'percentage_mov_status_cluster':
        # all_results = precisionSimga5(nusc, args.consecutive_frames, args.sigma)
        motion_categories, cf_indices = plot_mov_stationary_cluster(all_results)
        total_clusters_list, cf_indices = plot_num_cluster(all_results)
        plot_percentage_mov_status_cluster(motion_categories, total_clusters_list, cf_indices)
    elif args.plot_type == 'precistion_mov_status_cluster':
        # all_results = precisionSimga5(nusc, args.consecutive_frames, args.sigma)
        motion_categories, cf_indices = plot_mov_stationary_cluster(all_results)
        plot_precistion_mov_status_cluster(motion_categories, cf_indices)
    elif args.plot_type == 'rcs_cluster':
        # all_results = precisionSimga5(nusc, args.consecutive_frames, args.sigma)
        rcs_categories, cf_indices = plot_rcs_cluster(all_results, cf_indices)
    elif args.plot_type == 'percentage_rcs_status_cluster':
        # all_results = precisionSimga5(nusc, args.consecutive_frames, args.sigma)
        total_clusters_list, cf_indices = plot_num_cluster(all_results)
        rcs_categories, cf_indices = plot_rcs_cluster(all_results, cf_indices)
        plot_percentage_rcs_status_cluster(rcs_categories, total_clusters_list, cf_indices)
    elif args.plot_type == 'precision_rcs_status_cluster':
        # all_results = precisionSimga5(nusc, args.consecutive_frames, args.sigma)
        total_clusters_list, cf_indices = plot_num_cluster(all_results)
        rcs_categories, cf_indices = plot_rcs_cluster(all_results, cf_indices)
        plot_precision_rcs_status_cluster(rcs_categories, cf_indices)
    else:
        print("Invalid plot type. Choose from these choices; num_cluster, mov_stationary_cluster, \
                            percentage_mov_status_cluster, precistion_mov_status_cluster, \
                            rcs_cluster, percentage_rcs_status_cluster, precision_rcs_status_cluster")
    

if __name__ == "__main__":
    main()