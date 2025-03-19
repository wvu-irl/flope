"""
This scripts reads gt_poses.npy file and projects the gt poses to all the images
to check if it aligns well the real flowers.
Saves the outputs to gt_poses dir.
"""
import numpy as np
import cv2
from tqdm import tqdm

from sunflower.utils.io import (
    DatasetPath, read_intrinsics_yaml_to_K_h_w, get_files, pth,
    load_pose
)

if __name__=='__main__':
    data = DatasetPath('/home/rashik_shrestha/data/sunflower/plantscan_pixel_1230_raw')
    K,h,w = read_intrinsics_yaml_to_K_h_w(data.intrin)
    files = get_files(data.rgb)
    print(f"{len(files)} files available.")
    
    

    for file in tqdm(files):
        print(file)
        image = cv2.imread(pth(data.rgb,file,'jpg'))
        pose = load_pose(pth(data.pose,file,'txt')) # camera pose (not flower)
        extrinsics = np.linalg.inv(pose)
        print(extrinsics)
        exit()
        