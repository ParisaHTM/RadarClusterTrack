"""
Process scenes in the NuScenes dataset.
"""

import sys
import argparse
from config import DEFAULT_PATHS
from dataProcessor import RadarDataProcessor


def main():
    parser = argparse.ArgumentParser(description='Process NuScenes radar data for all scenes')
    parser.add_argument('--data-path', type=str, default=DEFAULT_PATHS['nuscenes_dataroot'], 
                       help='Path to NuScenes dataset')
    parser.add_argument('--start', type=int, default=0,
                       help='Starting scene index (default: 0)')
    parser.add_argument('--end', type=int, default=None,
                       help='Ending scene index (default: None for all remaining)')
    parser.add_argument('--scene-name', type=str, default=None,
                        help="Process a single scene by name")
    parser.add_argument('--output-dir', type=str, default=DEFAULT_PATHS['pickle_save_root'],
                       help='Output directory for results (optional)')
    
    args = parser.parse_args()
    
    config = {}
    if args.output_dir:
        config['pickle_save_root'] = args.output_dir
    

    processor = RadarDataProcessor(data_path=args.data_path, config=config)
    print("RadarDataProcessor initialized successfully!")

    available_scenes = processor.get_available_scenes()
    scene_names = list(available_scenes.keys())
    print(f"Found {len(available_scenes)} scenes in dataset")

    if args.scene_name:
        scenes_to_process = [args.scene_name]
        scene_range_desc = f"scene {args.scene_name}"
    elif args.end is None:
        scenes_to_process = scene_names[args.start:]
        scene_range_desc = f"scenes {args.start} to {len(available_scenes)-1}"
    else:
        scenes_to_process = [available_scenes[name] for name in scenes_to_process]
        scene_range_desc = f"scenes {args.start} to {args.end-1}"
    
    print(f"Will process {len(scenes_to_process)} scenes ({scene_range_desc})")
    
    if len(scenes_to_process) == 0:
        print("No scenes to process with the given range!")
        sys.exit(1)
    
    print(f"Scenes to process: {scenes_to_process[:5]}{'...' if len(scenes_to_process) > 5 else ''}")
    
    processor.process_scenes(scenes_to_process)
    
    print("All scenes processed successfully!")


if __name__ == "__main__":
    print("NuScenes Radar Data - Process All Scenes")
    print("=" * 40)
    main() 