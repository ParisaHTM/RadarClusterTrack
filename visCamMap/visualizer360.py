# visualizer.py

import os
import cv2
import pickle
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points
from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt
import os.path as osp
from visCamMap.config import CAMS, CAMERA_LABELS, DEFAULT_PATHS, VIDEO_PARAMS

class SceneVisualizer:
    def __init__(self, nusc):
        self.nusc = nusc
        self.data_dir = DEFAULT_PATHS["data_dir"]
        self.save_root = DEFAULT_PATHS["save_root"]
        self.individual_frames_root = DEFAULT_PATHS["individual_frames_root"]
        self.combined_frames_root = DEFAULT_PATHS["combined_frames_root"]
        self.fps = VIDEO_PARAMS["fps"]
        self.frame_size = VIDEO_PARAMS["frame_size"]

        os.makedirs(self.save_root, exist_ok=True)
        os.makedirs(self.individual_frames_root, exist_ok=True)
        os.makedirs(self.combined_frames_root, exist_ok=True)

    def load_clustering_data(self, scene_name):
        pickle_filename = os.path.join(self.data_dir, f"{scene_name}_clustering_data.pkl")
        if not os.path.exists(pickle_filename):
            print(f"File not found: {pickle_filename}")
            return None
        with open(pickle_filename, 'rb') as f:
            return pickle.load(f)

    def get_available_scenes(self):
        return {scene['name']: scene for scene in self.nusc.scene}

    def add_camera_label_and_frame(self, image, camera_name, frame_num):
        height, width = image.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale, thickness = 2.0, 4
        camera_label = CAMERA_LABELS.get(camera_name, camera_name)
        frame_text = f"Frame: {frame_num:04d}"

        (label_w, label_h), _ = cv2.getTextSize(camera_label, font, font_scale, thickness)
        (frame_w, frame_h), _ = cv2.getTextSize(frame_text, font, font_scale, thickness)
        overlay = image.copy()

        cv2.rectangle(overlay, (15, 15), (15 + label_w + 20, 15 + label_h + 20), (0, 0, 0), -1)
        cv2.rectangle(overlay, (width - frame_w - 30, 15),
                      (width - 10, 15 + frame_h + 20), (0, 0, 0), -1)

        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        cv2.putText(image, camera_label, (30, 60), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(image, frame_text, (width - frame_w - 20, 60), font, font_scale, (255, 255, 255), thickness)
        return image

    def visualize_scene(self, scene_name):
        data = self.load_clustering_data(scene_name)
        if not data:
            return

        total_frames = len(data) // len(CAMS)
        video_writer = cv2.VideoWriter(
            os.path.join(self.save_root, f"{scene_name}.mp4"),
            cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.frame_size
        )

        combined_frames_dir = os.path.join(self.combined_frames_root, scene_name)
        os.makedirs(combined_frames_dir, exist_ok=True)

        scene_camera_dirs = {}
        for cam in CAMS:
            cam_dir = os.path.join(self.individual_frames_root, scene_name, CAMERA_LABELS[cam].replace(' ', '_').lower())
            os.makedirs(cam_dir, exist_ok=True)
            scene_camera_dirs[cam] = cam_dir
        
        data_first = 0
        data_last = 6

        for frame_num in range(int(total_frames)):
            images_by_camera = {}
            for cam in CAMS:
                for camera_clustering_data in data[data_first:data_last]:
                    if camera_clustering_data['frame_num'] == frame_num and camera_clustering_data['camera'] == cam:
                        image_token = camera_clustering_data['sample_token']
                        sample_data = self.nusc.get('sample', image_token)
                        cam_sample_data = self.nusc.get('sample_data', sample_data['data'][cam])
                        im = Image.open(osp.join(DEFAULT_PATHS["data_path"], cam_sample_data['filename']))
                        image_np = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
                        poserecord_ego = self.nusc.get('ego_pose', cam_sample_data ['ego_pose_token'])
                        clusters = camera_clustering_data['clusters']
                        for cluster_num, cluster in enumerate(clusters):
                            points = camera_clustering_data['radar_points'][cluster_num][:3]
                            cluster_color_indices = camera_clustering_data['cluster_color_indices'][cluster_num]
                            cluster_unique_ids = camera_clustering_data['cluster_ids'][cluster_num]

                            points_ego_translated = np.zeros_like(points)
                            for j in range((points_ego_translated.shape[0])):
                                points_ego_translated[j,:] = points[j] - poserecord_ego["translation"][j]
                            points_ego_rotated = np.dot(Quaternion(poserecord_ego['rotation']).rotation_matrix.T, points_ego_translated)

                            cs_record_cam = self.nusc.get('calibrated_sensor', cam_sample_data['calibrated_sensor_token'])

                            points_cam = np.zeros_like(points_ego_rotated)
                            for j in range((points_cam.shape[0])):
                                points_cam[j] = points_ego_rotated[j] - cs_record_cam['translation'][j]

                            points_cam_rotated = np.dot(Quaternion(cs_record_cam['rotation']).rotation_matrix.T, points_cam)
                            
                            points_im = view_points(points_cam_rotated, np.array(cs_record_cam['camera_intrinsic']), normalize=True)
                            
                            depths = points_cam_rotated[2, :]
                            min_dist = 1.0
                            mask = np.ones(depths.shape[0], dtype=bool)
                            mask = np.logical_and(mask, depths > min_dist)
                            mask = np.logical_and(mask, points_im[0, :] > 1)
                            mask = np.logical_and(mask, points_im[0, :] < im.size[0] - 1)
                            mask = np.logical_and(mask, points_im[1, :] > 1)
                            mask = np.logical_and(mask, points_im[1, :] < im.size[1] - 1)
                            points = points_im[:, mask]
                            color_index = cluster_color_indices
                            colors = plt.cm.tab20(color_index % 20)  # Use tab20 colormap for more color variety
                            color_rgb = tuple(map(int, np.array(colors[:3]) * 255))
                            color_bgr = tuple(reversed(color_rgb))  # RGB to BGR for OpenCV
                            for px, py in zip(points[0, :], points[1, :]):
                                cv2.circle(image_np, (int(px), int(py)), 20, color_bgr, -1)
                            if len(points[0, :]) > 0:
                                center_x = int(np.mean(points[0, :]))
                                center_y = int(np.mean(points[1, :]))
                                real_cluster_id = cluster_unique_ids
                                text = f"ID:{real_cluster_id}"
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 3
                                thickness = 5
                                cv2.putText(image_np, text, (center_x-20, center_y-5), 
                                        font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                                cv2.putText(image_np, text, (center_x-20, center_y-5), 
                                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

                    image_np = self.add_camera_label_and_frame(image_np.copy(), cam, frame_num)
                    images_by_camera[cam] = cv2.resize(image_np, self.frame_size)

                    individual_frame_path = os.path.join(scene_camera_dirs[cam], f"frame_{frame_num:04d}.png")
                    cv2.imwrite(individual_frame_path, image_np)

            if len(images_by_camera) == len(CAMS):
                grid_top = np.hstack([
                    images_by_camera['CAM_FRONT_LEFT'],
                    images_by_camera['CAM_FRONT'],
                    images_by_camera['CAM_FRONT_RIGHT']
                ])
                grid_bottom = np.hstack([
                    images_by_camera['CAM_BACK_LEFT'],
                    images_by_camera['CAM_BACK'],
                    images_by_camera['CAM_BACK_RIGHT']
                ])
                grid_frame = np.vstack([grid_top, grid_bottom])
                combined_frame_path = os.path.join(combined_frames_dir, f"frame_{frame_num:04d}.png")
                cv2.imwrite(combined_frame_path, grid_frame)
                img_resize = cv2.resize(grid_frame, self.frame_size)
                video_writer.write(img_resize)
                data_first = data_last
                data_last = data_first + 6
                if data_last > len(data):
                    break
        video_writer.release()

    def visualize_scenes(self, scenes):
        for scene in scenes:
            print(f"Processing {scene}...")
            self.visualize_scene(scene)
