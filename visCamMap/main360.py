# main.py

import argparse
from nuscenes.nuscenes import NuScenes
from visCamMap.config import DEFAULT_PATHS
from visCamMap.visualizer360 import SceneVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize radar clustering scenes on camera view")
    parser.add_argument('--mode', choices=['single', 'range', 'all'], required=True,
                        help="Choose visualization mode: single, range, or all scenes")
    parser.add_argument('--scene', type=str, help="Scene name (for single mode)")
    parser.add_argument('--start', type=int, help="Start index (for range mode)")
    parser.add_argument('--end', type=int, help="End index (for range mode)")
    return parser.parse_args()

def main():
    args = parse_args()
    nusc = NuScenes(version="v1.0-mini", dataroot=DEFAULT_PATHS["data_path"], verbose=True)
    visualizer = SceneVisualizer(nusc)
    all_scenes = list(visualizer.get_available_scenes().keys())

    if args.mode == 'single':
        visualizer.visualize_scenes([args.scene])
    elif args.mode == 'range':
        scenes_to_process = all_scenes[args.start: args.end]
        visualizer.visualize_scenes(scenes_to_process)
    elif args.mode == 'all':
        visualizer.visualize_scenes(all_scenes)

if __name__ == "__main__":
    main()
