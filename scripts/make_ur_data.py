"""
Simple script to rename the files collected using UR5
"""

import os
from pathlib import Path
from tqdm import tqdm

dataset_dir = Path('/home/rashik_shrestha/data/andy_data/single_cluster')

rgb_files = os.listdir(dataset_dir/'rgb')
rgb_files.sort()

i=0
for rfile in tqdm(rgb_files):
    file = rfile[4:-4] 
    rgb_filename = dataset_dir/'rgb'/f"rgb_{file}.png"
    depth_filename = dataset_dir/'depth'/f"depth_{file}.png"
    poses_filename = dataset_dir/'pose'/f"pose_{file}.txt"
    
    if not rgb_filename.exists():
        print(rgb_filename, 'not exist')
        continue
    if not depth_filename.exists():
        print(depth_filename, 'not exist')
        continue
    if not poses_filename.exists():
        print(poses_filename, 'not exist')
        continue
    
    os.rename(rgb_filename, dataset_dir/'rgb'/f"frame_{i:05d}.png")
    os.rename(depth_filename, dataset_dir/'depth'/f"frame_{i:05d}.png")
    os.rename(poses_filename, dataset_dir/'pose'/f"frame_{i:05d}.txt")
    i+=1
    
    #TODO: remove unnamed images
    # os.rename(dataset_dir/'detection'/f"rgb_{file}.txt", dataset_dir/'detection'/f"frame_{i:05d}.txt")
   
print('done') 