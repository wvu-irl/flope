import cv2
import numpy as np
from tyro import cli
from pathlib import Path
import os
from tqdm import tqdm

from sunflower.utils.image_manipulation import detection_and_mask_to_contours, contours_to_polygons
from sunflower.utils.plot import plot_bounding_boxes

def main(
    data_dir: str = '/users/rs00140/dataset/yolo_seg',
    split: str = 'train',
    idx0: int = 0,
    idx1: int = -1,
):
    data_dir = Path(data_dir)
    mask_dir = data_dir/'masks'/split
    det_dir = data_dir/'dets'/split
    label_dir = data_dir/'labels'/split
    img_dir = data_dir/'images'/split
    
    files = os.listdir(mask_dir)
    files.sort()
    files = files[idx0:idx1]
    print(f"{len(files)} files available.")
    
    for file in tqdm(files):
        try:
            print(file)
            mask = cv2.imread(mask_dir/file, cv2.IMREAD_GRAYSCALE)
            bbox = np.loadtxt(det_dir/f"{file[:-3]}txt")
            
            H,W = mask.shape
            contours = detection_and_mask_to_contours(mask, bbox)
            polygons = contours_to_polygons(contours, H, W)

            # Draw contours for debugging
            # img = cv2.imread(img_dir/file)
            # cv2.drawContours(img, contours, -1, (0, 0, 255), thickness=10)
            # cv2.drawContours(img, contours, -1, (0, 0, 255), thickness=10)
            # cv2.imwrite('contours.png', img)
            
            with open(label_dir/f"{file[:-3]}txt", 'w') as fp:
                for poly in polygons:
                    line = " ".join(map(str, poly))
                    fp.write("0 " + line + "\n")  # Add a newline after each sublist
        except:
            print(f"error in file: {file}")
        
    
if __name__=='__main__':
    cli(main)
