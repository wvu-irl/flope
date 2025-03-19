"""
This script was written to simple detect flowers in a set of flowerur data.
In order to back project them to 3D using depth map
"""

from PIL import Image
import cv2
import numpy as np
import os
from sunflower.models.grounding_dino import GroundingDINO
from pathlib import Path
from tqdm import tqdm

device = 'cuda' 
text_prompt = 'white flower.'
img_dir = Path('/home/rashik_shrestha/data/flowerur/nobackdrop_high/rgb')
out_dir = Path('/home/rashik_shrestha/data/flowerur/nobackdrop_high/detection')
out_dir.mkdir(exist_ok=True)
print("defining gdino")
gdino = GroundingDINO(device, text_prompt, box_th=0.3, text_th=0.3)
print('done defining')

#! Files
files = os.listdir(img_dir)
files.sort()

for file in tqdm(files):
    #! Read image
    img_pil = Image.open(img_dir/file) # PIL (RGB)
    img_cv = np.array(img_pil) # OpenCV RGB
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR) # OpenCV BGR

    #! Detection
    bb_dino = gdino.detect(img_cv)
    bb_dino = np.array(bb_dino)
    np.savetxt(out_dir/f"{file[:-3]}txt", bb_dino)
    # print(bb_dino)
    # print(bb_dino.shape)
    # exit()
    
