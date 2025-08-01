from nuscenes.nuscenes import NuScenesExplorer
from nuscenes.utils.geometry_utils import view_points, BoxVisibility, transform_matrix
from typing import Tuple, List, Iterable
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import math
import numpy as np
import copy
from functools import reduce
from PIL import Image

class NuScenesExplorerOffRoadPoints(NuScenesExplorer):

    def __init__(self, nusc):
        super().__init__(nusc)
    axes_limit = 150

    def get_cluster_color(self, cluster_id):
        """
        Generate a consistent color for each cluster ID.
        """
        import matplotlib.pyplot as plt
        cmap = plt.cm.tab20  
        color_idx = cluster_id % 20
        return cmap(color_idx)
    
    def find_off_road_points(self,
                     sd_token: str,
                     frame_data,
                      out_path: str = None,
                      verbose: bool = True,
                      frame_num: int = None,
                      cluster_labels = None):
        """
        Find all off-road radar points cloud in a sample.
        :param sd_token: Sample data token.
        :param frame_data: Frame data containing clustering information.
        # :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        # :param nsweeps: Number of sweeps for lidar and radar.
        :param out_path: Optional path to save the rendered figure to disk.
        # :param show_lidarseg: Whether to show lidar segmentations labels or not.
        # :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes.
        # :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
        #                                 predictions for the sample.
        :param verbose: Whether to show the rendered sample in a window or not.
        # :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
        #     to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        #     If show_lidarseg is True, show_panoptic will be set to False.
        """
        fig, axes = plt.subplots(1, 1, figsize=(16, 16))  
        fig.patch.set_facecolor('black') 

        if isinstance(axes, plt.Axes):
            axes = np.array([[axes]])
        elif axes.ndim == 1:
            axes = axes.reshape(1, -1)

        ax = axes[0, 0]
        info_points = self.sample_radar_data_on_map(sd_token, 
                                                    frame_data,
                                                    ax=ax, 
                                                    verbose=False,
                                                    cluster_labels=cluster_labels)
        
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        
        show_title = True  
        if show_title:
            scene_name = frame_data[0]['scene_name'] if frame_data else "Unknown"
            title_text = f'{scene_name}'
            if frame_num is not None:
                title_text += f' - Frame {frame_num}'
            
            ax.text(0.02, 0.98, title_text, transform=ax.transAxes, 
                   fontsize=20, fontweight='bold', color='white',
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
        
        if verbose:
            if out_path is not None:
                plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0, 
                           facecolor='black', edgecolor='none')
            plt.show()
        
        return info_points, fig
    
    def sample_radar_data_on_map(self,
                            sample_data_token: str,
                            frame_data,
                            axes_limit: int = axes_limit,
                            ax: Axes = None,
                            out_path: str = None,
                            underlay_map: bool = True,
                            use_flat_vehicle_coordinates: bool = True,
                            verbose: bool = True,
                            cluster_labels=None,
                            show_legend: bool = False,
                           ):
        """
        Render sample data onto axis.
        :param sample_data_token: Sample_data token.
        :param axes_limit: Axes limit for lidar and radar (measured in meters).
        :param ax: Axes onto which to render.
        :param out_path: Optional path to save the rendered figure to disk.
        :param underlay_map: When set to true, lidar data is plotted onto the map. This can be slow.
        :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
            aligned to z-plane in the world. Note: Previously this method did not use flat vehicle coordinates, which
            can lead to small errors when the vertical axis of the global frame and lidar are not aligned. The new
            setting is more correct and rotates the plot by ~90 degrees.
        :param verbose: Whether to display the image after it is rendered.
        """

        sd_record = self.nusc.get('sample_data', sample_data_token)
        sensor_modality = sd_record['sensor_modality']
        sample_token = frame_data[0]['sample_token']  
        if sensor_modality =='radar':
            sample_rec = self.nusc.get('sample', sample_token)
            # chan = sd_record['channel']
            ref_chan = 'LIDAR_TOP'
            ref_sd_token = sample_rec['data'][ref_chan]
            ref_sd_record = self.nusc.get('sample_data', ref_sd_token)
            ref_pose_rec = self.nusc.get('ego_pose', ref_sd_record ['ego_pose_token'])
            ref_cs_rec = self.nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
            ego_rotation = Quaternion(ref_pose_rec['rotation'])
            yaw = ego_rotation.yaw_pitch_roll[0]
            # Homogeneous transform from ego car frame to reference frame.
            ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)

            # Homogeneous transformation matrix from global to _current_ ego car frame.
            car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                            inverse=True)
            trans_matrix_lidar = reduce(np.dot, [ref_from_car, car_from_global,])

            all_cluster_points = []
            cluster_colors = []
            cluster_ids = []
            for camera_data in frame_data:
                clusters = camera_data['clusters']
                cluster_id_list = camera_data['cluster_ids']
                x_coords = camera_data['cluster_global_positions']['x']
                y_coords = camera_data['cluster_global_positions']['y']
                
                for cluster_indices, cluster_id in zip(clusters, cluster_id_list):
                    if len(cluster_indices) > 0:  
                        cluster_x = x_coords[cluster_indices]
                        cluster_y = y_coords[cluster_indices]
                        
                        all_cluster_points.extend(list(zip(cluster_x, cluster_y)))
                        cluster_colors.extend([self.get_cluster_color(cluster_id)] * len(cluster_indices))
                        cluster_ids.extend([cluster_id] * len(cluster_indices))

            # Convert to numpy array format compatible with point cloud operations
            all_cluster_array = np.array(all_cluster_points).T  # Shape: (2, N) for x,y coordinates
            # Add z=0 and dynamic property placeholder
            zeros = np.zeros((1, all_cluster_array.shape[1]))
            cluster_ids_array = np.array(cluster_ids).reshape(1, -1)
            all_cluster_points_pc = np.vstack([all_cluster_array, zeros, cluster_ids_array])  # Shape: (4, N)
            
            all_radar_before_trans = copy.deepcopy(all_cluster_points_pc)
            
            # Transform points to lidar coordinate system
            homogeneous_points = np.vstack([all_cluster_points_pc[:3, :], np.ones((1, all_cluster_points_pc.shape[1]))])
            transformed_points = trans_matrix_lidar @ homogeneous_points
            all_cluster_points_pc[:3, :] = transformed_points[:3, :]


            # By default we render the sample_data top down in the sensor frame.
            # This is slightly inaccurate when rendering the map as the sensor frame may not be perfectly upright.
            # Using use_flat_vehicle_coordinates we can render the map in the ego frame instead.
            if use_flat_vehicle_coordinates:
                # Retrieve transformation matrices for reference point cloud.
                cs_record = self.nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
                pose_record = self.nusc.get('ego_pose', ref_sd_record['ego_pose_token'])
                ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                              rotation=Quaternion(cs_record["rotation"]))

                # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
                ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
                rotation_vehicle_flat_from_vehicle = np.dot(
                    Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
                    Quaternion(pose_record['rotation']).inverse.rotation_matrix)
                vehicle_flat_from_vehicle = np.eye(4)
                vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
                viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)
            else:
                viewpoint = np.eye(4)


            points = view_points(all_cluster_points_pc[:3, :], viewpoint, normalize=False)
            dyn_prop = all_cluster_points_pc[3, :].astype(int)
            
            # Render map if requested.
            if underlay_map:
                assert use_flat_vehicle_coordinates, 'Error: underlay_map requires use_flat_vehicle_coordinates, as ' \
                                                     'otherwise the location does not correspond to the map!'
                ego_map = self.ego_centric_map(sample_data_token=sample_data_token, axes_limit=axes_limit, ax=ax, plot=True)
                off_road_mask = self.detect_off_road(points[:3, :], ego_map, axes_limit)
                dyn_prop[off_road_mask] = 8
                all_radar_before_trans[3, :] = dyn_prop
                
                cluster_point_counts = {}
                cluster_offroad_counts = {}
                
                for i in range(len(cluster_ids)):
                    cluster_id = cluster_ids[i]
                    if cluster_id not in cluster_point_counts:
                        cluster_point_counts[cluster_id] = 0
                        cluster_offroad_counts[cluster_id] = 0
                    cluster_point_counts[cluster_id] += 1
                    if off_road_mask[i]:
                        cluster_offroad_counts[cluster_id] += 1
                
                # off_road_cluster_ids = set()
                for i in range(len(cluster_colors)):
                    if off_road_mask[i]:
                        cluster_colors[i] = (0.0, 0.0, 0.0, 1.0)  # Black color for off-road points

            colors = np.array(cluster_colors)
            
            point_scale = 70.0  # Made points bigger
            ax.scatter(points[0, :], points[1, :], c=colors, s=point_scale)
            
            cluster_centers = {}
            cluster_point_counts = {}
            
            for i, cluster_id in enumerate(cluster_ids):
                if cluster_id not in cluster_centers:
                    cluster_centers[cluster_id] = [0, 0]
                    cluster_point_counts[cluster_id] = 0
                cluster_centers[cluster_id][0] += points[0, i]
                cluster_centers[cluster_id][1] += points[1, i]
                cluster_point_counts[cluster_id] += 1
            
            for cluster_id in cluster_centers:
                if cluster_point_counts[cluster_id] > 0:
                    center_x = cluster_centers[cluster_id][0] / cluster_point_counts[cluster_id]
                    center_y = cluster_centers[cluster_id][1] / cluster_point_counts[cluster_id]
                    label_text = str(cluster_id)   
                    fontsize = 18
                    bbox_style = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='black')
                    text_color = 'black'
                
                    ax.text(center_x, center_y, label_text, 
                           fontsize=fontsize, fontweight='bold', 
                           ha='center', va='center',
                           bbox=bbox_style,
                           color=text_color)

            ego_x, ego_y = 0, 0  
            
            pose_record = self.nusc.get('ego_pose', ref_sd_record['ego_pose_token'])
            ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            
            arrow_length = 10 
            arrow_dx = np.cos(ego_yaw) * arrow_length
            arrow_dy = np.sin(ego_yaw) * arrow_length
            
            ax.plot(ego_x, ego_y, 'o', color='red', markersize=12, markeredgecolor='black', markeredgewidth=2)
            
        ax.axis('off')
        ax.set_aspect('equal')
        
        if show_legend:
            unique_cluster_ids = list(set(cluster_ids))
            legend_elements = []
            for cluster_id in unique_cluster_ids:
                if cluster_labels and cluster_id in cluster_labels:
                    label_text = f"{cluster_labels[cluster_id]}"
                else:
                    label_text = f"Cluster {cluster_id}"
                legend_elements.append(
                    Patch(facecolor=self.get_cluster_color(cluster_id), label=label_text)
                )
            ax.legend(handles=legend_elements, loc='upper right', fontsize='small', 
                     frameon=True, fancybox=True, shadow=True, framealpha=0.9)

        if out_path is not None:
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)

        if verbose:
            plt.show()
        
        return all_radar_before_trans
    
    def ego_centric_map(self,
                        sample_data_token: str,
                        axes_limit: float = axes_limit,
                        ax: Axes = None,
                        plot: bool = True):
        """
        Render map centered around the associated ego pose.
        :param sample_data_token: Sample_data token.
        :param axes_limit: Axes limit measured in meters.
        :param ax: Axes onto which to render.
        """

        def crop_image(image: np.array,
                       x_px: int,
                       y_px: int,
                       axes_limit_px: int) -> np.array:
            x_min = int(x_px - axes_limit_px)
            x_max = int(x_px + axes_limit_px)
            y_min = int(y_px - axes_limit_px)
            y_max = int(y_px + axes_limit_px)

            cropped_image = image[y_min:y_max, x_min:x_max]

            return cropped_image

        # Get data.
        sd_record = self.nusc.get('sample_data', sample_data_token)
        sample = self.nusc.get('sample', sd_record['sample_token'])
        scene = self.nusc.get('scene', sample['scene_token'])
        log = self.nusc.get('log', scene['log_token'])
        map_ = self.nusc.get('map', log['map_token'])
        map_mask = map_['mask']
        pose = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
        ego_rotation = Quaternion(pose['rotation'])
        yaw = ego_rotation.yaw_pitch_roll[0]

        # Retrieve and crop mask.
        pixel_coords = map_mask.to_pixel_coords(pose['translation'][0], pose['translation'][1])
        scaled_limit_px = int(axes_limit * (1.0 / map_mask.resolution))
        mask_raster = map_mask.mask()
        cropped = crop_image(mask_raster, pixel_coords[0], pixel_coords[1], int(scaled_limit_px * math.sqrt(2)))

        # Rotate image.
        ypr_rad = Quaternion(pose['rotation']).yaw_pitch_roll
        yaw_deg = -math.degrees(ypr_rad[0])
        rotated_cropped = np.array(Image.fromarray(cropped).rotate(yaw_deg))

        # Crop image.
        ego_centric_map = crop_image(rotated_cropped,
                                     int(rotated_cropped.shape[1] / 2),
                                     int(rotated_cropped.shape[0] / 2),
                                     scaled_limit_px)
        # Init axes and show image.
        # Set background to white and foreground (semantic prior) to gray.
        ego_centric_map[ego_centric_map == map_mask.foreground] = 125
        ego_centric_map[ego_centric_map == map_mask.background] = 255
        ax.imshow(ego_centric_map, extent=[-axes_limit, axes_limit, -axes_limit, axes_limit],
                cmap='gray', vmin=0, vmax=255)
        
        # Return map mask and metadata needed for coordinate alignment
        return ego_centric_map # center_x, center_y are always in center

    def detect_off_road(self, points, ego_centric_map, axes_limit):
        """
        Given radar points and an ego-centric map, return a mask indicating on-road points.
        Points are considered on-road if they are on gray sections (value 125) in ego_centric_map.
        """
        map_height, map_width = ego_centric_map.shape
        
        x_px = ((points[0] + axes_limit) * map_width / (2 * axes_limit)).astype(int)
        y_px = ((axes_limit - points[1]) * map_height / (2 * axes_limit)).astype(int)
        
        x_px = np.clip(x_px, 0, map_width - 1)
        y_px = np.clip(y_px, 0, map_height - 1)
        
        off_road_mask = ego_centric_map[y_px, x_px] == 255     
        
        return off_road_mask