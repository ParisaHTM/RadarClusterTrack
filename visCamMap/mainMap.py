# main.py

import argparse
from nuscenes.nuscenes import NuScenes
from config import DEFAULT_PATHS
from nusceneExploreOffRoad import NuScenesExplorerOffRoadPoints
from visualizerMap import *

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize radar clustering scenes on global map")
    parser.add_argument('--mode', choices=['single', 'range', 'all'], required=True,
                        help="Choose visualization mode: single, range, or all scenes")
    parser.add_argument('--scene', type=str, help="Scene name (for single mode)")
    parser.add_argument('--start', type=int, help="Start index (for range mode)")
    parser.add_argument('--end', type=int, help="End index (for range mode)")
    return parser.parse_args()

def main():
    args = parse_args()
    nusc = NuScenes(version="v1.0-mini", dataroot=DEFAULT_PATHS["data_path"], verbose=True)
    explorer = NuScenesExplorerOffRoadPoints(nusc)
    all_scenes = list(get_available_scenes(nusc).keys())

    if args.mode == 'single':
        visualize_global_map_all(nusc,[args.scene], explorer)
    elif args.mode == 'range':
        scenes_to_process = all_scenes[args.start: args.end]
        visualize_global_map_all(nusc,scenes_to_process, explorer)
    elif args.mode == 'all':
        visualize_global_map_all(nusc,all_scenes, explorer)

if __name__ == "__main__":
    main()
