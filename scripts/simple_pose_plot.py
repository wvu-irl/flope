from sunflower.predictor.pose_predictor import PosePredictor
import cv2
from sunflower.utils.plot import plot_flower_poses_on_image
from sunflower.utils.io import read_intrinsics_yaml_to_K_h_w
import numpy as np
import os 
from pathlib import Path
from tqdm import tqdm

#! Params
# rgb_path = '/home/rashik_shrestha/data/sunflower/plantscan_pixel_1230_raw/rgb/frame_00017.jpg'
# depth_path = '/home/rashik_shrestha/data/sunflower/plantscan_pixel_1230_raw/depth_png/frame_00017.png'
# intrin_path = '/home/rashik_shrestha/data/sunflower/plantscan_pixel_1230_raw/intrinsics.yaml'
# intrin_path = '/home/rashik_shrestha/data/sunflower/flowerur/intrinsics.yaml'

rgb_path = '/home/rashik_shrestha/ws/sunflower/output/r405_near_rgb.png'
depth_path = '/home/rashik_shrestha/ws/sunflower/output/r405_near_depth.png'
intrin_path = '/home/rashik_shrestha/data/sunflower/flower_r405/intrinsics.yaml'
out_path = 'r405_far_poses.png'


model = PosePredictor(
    device='cuda',
    posenet_path='/home/rashik_shrestha/ws/sunflower/scripts/weights/posenet_e183.pth',
    intrin_path=intrin_path,
    debug=True
)
K,h,w = read_intrinsics_yaml_to_K_h_w(intrin_path)

rgb = cv2.imread(rgb_path)
depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

flower_pose = model.get_flower_poses(rgb, depth)

if flower_pose is not None:
    print(f"Found {flower_pose.shape} flowers.")
    plot_flower_poses_on_image(rgb, flower_pose, K)
else:
    print("No flowers detected 🙁")
    
cv2.imwrite(out_path, rgb)