import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

#! Check input params
if len(sys.argv) < 2:
    print("Usage: python undistort_images.py /path/to/input/dir /path/to/output/dir")
    exit()

#! Inputs
input_dir = sys.argv[1]
output_dir = sys.argv[2]
# input_dir = '/home/rashik_shrestha/data/flower_scan_raw'
# output_dir = '/home/rashik_shrestha/data/flower_scan_undistort'

#! Paths
input_dir = Path(input_dir)
output_dir = Path(output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

#! Camera Intrinsics for Pixel6a
camera_matrix = np.array([[1751.27658,    0.     ,  957.98419],
                          [0.     , 1756.38916,  529.39339],
                          [0.     ,    0.     ,    1.     ]], dtype=np.float32)

dist_coeffs = np.array([0.113261, -0.330971, -0.000624, -0.002176, 0.000000], dtype=np.float32)

#! Get images path recursively
images = [file.relative_to(input_dir) for file in input_dir.rglob("*.jpg")]
images.sort()

#! Undistort images 
for img in tqdm(images):
    image = cv2.imread(input_dir/img)
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)
    (output_dir/img).parent.mkdir(parents=True, exist_ok=True) # Make output dir if doesn't exist
    cv2.imwrite(output_dir/img, undistorted_image) # Write Undistorted image

print("Done :)")