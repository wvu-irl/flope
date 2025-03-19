"""
Nerfstudio tool ns-train needs data to be in specific format. This data
preparation can also be done using ns-process-data. sai-cli can also prepare 
data in this format.

In the nerfstudio format, camera poses (and multiple other stuffs) are stored
in transforms.json file.

This script reads this json file and extract camera poses and stores them in
data_dir/poses directory.

4x4 pose matrix is stored in json_data->frames->transform_matrix
pose matrix is in this form:
[R|t]
[0|1]

Output to file with 12 dim vector.
First 9 = rotmat
Last 3 = translation
"""

import json
import numpy as np
from pathlib import Path

from sunflower.utils.conversion import openGL_to_openCV_c2w

#! Parameters
transforms_file = "/home/rashik_shrestha/ws/splats/plantscan_pixel_1230/sai/transforms.json"
data_dir = '/home/rashik_shrestha/data/sunflower/plantscan_pixel_1230_raw'

data_dir = Path(data_dir)

with open(transforms_file, "r") as json_file:
    data = json.load(json_file)

print(len(data['frames']), "frames available")

#! Get dataparser_transforms
# splat_transforms_pth = '/home/rashik_shrestha/outputs/plantscan_pixel_1230_sai/splatfacto/2024-12-30_190922/dataparser_transforms.json'
# with open(splat_transforms_pth, 'r') as f:
#     splat_tf = json.load(f)
    
# splat_Rt = np.array(splat_tf['transform']) # Gives 3x4 mat
# splat_Rt = np.vstack((splat_Rt, np.array([0,0,0,1]))) # Get 4x4 mat
# splat_scale = splat_tf['scale']

for frame in data['frames']:
    name = frame['file_path'].split('/')[-1][:-4]
    
    pose = np.array(frame['transform_matrix'])
    
    #! Use dataparser transforms
    # pose = pose@splat_Rt
    
    pose = openGL_to_openCV_c2w(pose) 
    R = pose[:3,:3]
    
    #! Rotate to potrait
    # neg_90_degree_z_axis_rotation = np.array([
    #     [  0.0000000,  1.0000000,  0.0000000],
    #     [-1.0000000,  0.0000000, 0.0000000],
    #     [0.0000000,  0.0000000,  1.0000000 ]
    # ])
    # R = R@neg_90_degree_z_axis_rotation
    
    R = R.flatten()
    t = pose[:3,3]
    # t = pose[:3,3] * splat_scale # Use data transform scaling
    data = np.hstack((R,t))
    
    np.savetxt(data_dir/f"pose/{name}.txt", data)