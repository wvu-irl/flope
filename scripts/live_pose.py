from sunflower.predictor.pose_predictor import PosePredictor
import cv2
from sunflower.utils.plot import plot_flower_poses_on_image
from sunflower.utils.io import read_intrinsics_yaml_to_K_h_w
import numpy as np
import os 
from pathlib import Path
from tqdm import tqdm

#! Params
data_path = '/home/rashik_shrestha/data/flower_lib'
intirin_path = '/home/rashik_shrestha/data/sunflower/flowerur/intrinsics.yaml'

#! Get files
data_path = Path(data_path)
images = os.listdir(data_path/'rgb')
images.sort()
print(f"{len(images)} files available.")
out_path = data_path/'detection'
out_path.mkdir(exist_ok=True, parents=True)

model = PosePredictor(
    device='cuda',
    # posenet_path='/home/rashik_shrestha/ws/sunflower/scripts/weights/posenet_e183.pth',
    posenet_path='/home/rashik_shrestha/ws/sunflower/scripts/weights_part2/posenet_e104.pth',
    intrin_path=intirin_path,
    debug=True
)
K,h,w = read_intrinsics_yaml_to_K_h_w(intirin_path)

#! Read inputs
for img in tqdm(images):
    rgb = cv2.imread(data_path/'rgb'/img)
    depth = cv2.imread(data_path/'depth'/img, cv2.IMREAD_UNCHANGED)

    flower_pose = model.get_flower_poses(rgb, depth)
    
    if flower_pose is not None:
        plot_flower_poses_on_image(rgb, flower_pose, K)
        
    cv2.imwrite(data_path/'detection'/img, rgb)