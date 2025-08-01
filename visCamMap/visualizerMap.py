import pickle
import os
import cv2
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from config import DEFAULT_PATHS

def get_available_scenes(nusc):
     return {scene['name']: scene for scene in nusc.scene}

def load_clustering_data(scene_name, pickle_data_dir=DEFAULT_PATHS['data_dir'] ):
    """
    Load clustering data for a specific scene
    """
    pickle_filename = os.path.join(pickle_data_dir, f"{scene_name}_clustering_data.pkl")
    
    if not os.path.exists(pickle_filename):
        print(f"File not found: {pickle_filename}")
        return None
    
    with open(pickle_filename, 'rb') as f:
        data = pickle.load(f)
    
    return data

def create_video_from_frames(scene_name, frame_images, output_dir=DEFAULT_PATHS["global_map_root"], fps=5):
    """
    Create a video from a list of frame images
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    scene_output_dir = os.path.join(output_dir, scene_name)
    if not os.path.exists(scene_output_dir):
        os.makedirs(scene_output_dir)
    
    for i, img_array in enumerate(frame_images):
        frame_path = os.path.join(scene_output_dir, f"frame_{i:04d}.png")
        cv2.imwrite(frame_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    
    video_path = os.path.join(output_dir, f"{scene_name}_cluster_map.mp4")
    
    if len(frame_images) > 0:
        height, width, layers = frame_images[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        for img_array in frame_images:
            video.write(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        
        video.release()
        print(f"Video saved: {video_path}")
        print(f"Frames saved in: {scene_output_dir}")
    
    return video_path

def fig_to_array(fig):
    """
    Convert matplotlib figure to numpy array
    """
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(canvas.get_width_height()[::-1] + (3,))
    return buf

def analyze_cluster_movement_and_position(frames_data, nusc):
    """
    Analyze cluster movement patterns and road positions to assign semantic labels
    cluster_tracks = {frame_num: {cluster_id: {
        'positions': [],
        'stretches': [],
        'first_frame': frame_num,
        'last_frame': frame_num
    }}}
    """
    cluster_tracks = {}
    cluster_labels = {}
    
    for frame_num, frame_entries in frames_data.items():
        cluster_tracks[frame_num] = {}
        cluster_labels[frame_num] = {}
        for camera_data in frame_entries:
            clusters = camera_data['clusters']
            cluster_id_list = camera_data['cluster_ids']
            x_coords = camera_data['cluster_global_positions']['x']
            y_coords = camera_data['cluster_global_positions']['y']
            
            for cluster_indices, cluster_id in zip(clusters, cluster_id_list):
                if len(cluster_indices) > 0:
                    cluster_x = x_coords[cluster_indices]
                    cluster_y = y_coords[cluster_indices]
                    
                    if cluster_id not in cluster_tracks[frame_num]:
                        cluster_tracks[frame_num][cluster_id] = {
                            'positions': [],
                            'stretches': [],
                            'first_frame': frame_num,
                            'last_frame': frame_num
                        }
                    
                    cluster_tracks[frame_num][cluster_id]['positions'].append({
                        'center_x': np.mean(cluster_x),
                        'center_y': np.mean(cluster_y),
                        'points_x': cluster_x,
                        'points_y': cluster_y
                    })
                    cluster_tracks[frame_num][cluster_id]['last_frame'] = frame_num
                    
                    stretch_x = np.std(cluster_x) if len(cluster_x) > 1 else 0
                    stretch_y = np.std(cluster_y) if len(cluster_y) > 1 else 0
                    cluster_tracks[frame_num][cluster_id]['stretches'].append({
                        'stretch_x': stretch_x,
                        'stretch_y': stretch_y,
                        'max_stretch': max(stretch_x, stretch_y)
                    })
    
    all_movements = []
    all_stretch_x_list = []
    all_stretch_y_list = []
    all_ratio_stretch = []
    for frame_num, cluster_ids in cluster_tracks.items():
        for cluster_id, track_data in cluster_ids.items():
            first_pos = track_data['positions'][0]
            last_pos = track_data['positions'][-1]
            delta_x = last_pos['center_x'] - first_pos['center_x']
            delta_y = last_pos['center_y'] - first_pos['center_y']
            total_movement = np.sqrt(delta_x**2 + delta_y**2)
            all_movements.append(total_movement)
              
            stretch_values = track_data['stretches']
            all_stretch_x = sum(item['stretch_x'] for item in stretch_values)
            all_stretch_y = sum(item['stretch_y'] for item in stretch_values)
            ratio_stretch = max(all_stretch_x/all_stretch_y, all_stretch_y/all_stretch_x)
            all_stretch_x_list.append(all_stretch_x)
            all_stretch_y_list.append(all_stretch_y)
            all_ratio_stretch.append(ratio_stretch)
        
    
        min_movement = min(all_movements)
        max_movement = max(all_movements)
        avg_movement = sum(all_movements) / len(all_movements)

        for cluster_id, track_data in cluster_ids.items():
            first_pos = track_data['positions'][0]
            last_pos = track_data['positions'][-1]
            
            delta_x = last_pos['center_x'] - first_pos['center_x']
            delta_y = last_pos['center_y'] - first_pos['center_y']
            total_movement = np.sqrt(delta_x**2 + delta_y**2)

            stretch_values = track_data['stretches']
            all_stretch_x = sum(item['stretch_x'] for item in stretch_values)
            all_stretch_y = sum(item['stretch_y'] for item in stretch_values)
            ratio_stretch = max(all_stretch_x/all_stretch_y, all_stretch_y/all_stretch_x)


            ratio_threshold_low = min(all_ratio_stretch) + (max(all_ratio_stretch) - min(all_ratio_stretch)) * 0.1
            ratio_threshold_med = min(all_ratio_stretch) + (max(all_ratio_stretch) - min(all_ratio_stretch)) * 0.5
            ratio_threshold_high = min(all_ratio_stretch) + (max(all_ratio_stretch) - min(all_ratio_stretch)) * 0.7
            ratio_threshold_max = max(all_ratio_stretch)
            ratio_threshold_min = min(all_ratio_stretch)

            movement_threshold_low = min_movement + (max_movement - min_movement) * 0.1   # 10% of max movement
            movement_threshold_med = min_movement + (max_movement - min_movement) * 0.5   # 40% of max movement
            movement_threshold_high = min_movement + (max_movement - min_movement) * 0.7  # 70% of max movement

            stretch_threshold = 3.0  # meters

            if total_movement <= movement_threshold_low:
                cluster_labels[frame_num][cluster_id] = 'SO'

            elif movement_threshold_high <= total_movement <= max_movement:
                cluster_labels[frame_num][cluster_id] = 'MC'

            elif ratio_threshold_med < ratio_stretch <= ratio_threshold_max:
                if movement_threshold_med <= total_movement < movement_threshold_high:
                    cluster_labels[frame_num][cluster_id] = 'MC'
                else:  # movement_threshold_low < total_movement < movement_threshold_med
                    cluster_labels[frame_num][cluster_id] = 'SO'

            elif ratio_stretch > ratio_threshold_low:
                if total_movement > movement_threshold_low:
                    cluster_labels[frame_num][cluster_id] = 'MP'
                else:
                    cluster_labels[frame_num][cluster_id] = 'SO'

            elif ratio_stretch < ratio_threshold_low:
                if   movement_threshold_low < total_movement < movement_threshold_med:
                    cluster_labels[frame_num][cluster_id] = 'MO'
                else:
                    cluster_labels[frame_num][cluster_id] = 'SO'
            else:
                cluster_labels[frame_num][cluster_id] = 'SO'
    
    return cluster_labels

def visualize_global_map(nusc, scene_name, explorer):
    data = load_clustering_data(scene_name)
    if data is None:
        print(f"No data found for {scene_name}")

    frames = defaultdict(list)
    for entry in data:
        frames[entry['frame_num']].append(entry)
    
    cluster_labels = analyze_cluster_movement_and_position(frames, nusc)
    frame_images = []

    for frame_num in sorted(frames.keys()):
        frame_entries = frames[frame_num]         
        sample_token  = frame_entries[0]['sample_token']
        sample_rec = nusc.get('sample', sample_token)
        radar_sd_token = sample_rec['data']['RADAR_FRONT'] 
        
        try:
            info_points, fig = explorer.find_off_road_points(radar_sd_token,
                                                           frame_entries,
                                                           verbose=False,
                                                           frame_num=frame_num,
                                                           cluster_labels=cluster_labels[frame_num])
            img_array = fig_to_array(fig)
            frame_images.append(img_array)
            plt.close(fig)
            
        except Exception as e:
            print(f"Error processing frame {frame_num}: {e}")
            continue

    if frame_images:
        video_path = create_video_from_frames(scene_name, frame_images, fps=5)
        print(f"Created video for {scene_name}: {video_path}")
    else:
        print(f"No frames collected for {scene_name}")
    
    print(f"Completed processing {scene_name}")

def visualize_global_map_all(nusc, scene_names, explorer):
    for scene_name in scene_names:
        print(f"Processing {scene_name}...")
        visualize_global_map(nusc, scene_name, explorer)