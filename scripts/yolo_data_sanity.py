import os
from tqdm import tqdm
from pathlib import Path

images_dir = Path("/scratch/rs00140/dataset/yolo_seg/images/val")
masks_dir = Path("/scratch/rs00140/dataset/yolo_seg/masks/val")
dets_dir = Path("/scratch/rs00140/dataset/yolo_seg/dets/val")
labels_dir = Path("/scratch/rs00140/dataset/yolo_seg/labels/val")

files = os.listdir(images_dir)
files.sort()

count = 0
for file in tqdm(files):
    image_file = images_dir/file
    mask_file = masks_dir/file
    det_file = dets_dir/f"{file[:-3]}txt"
    label_file = labels_dir/f"{file[:-3]}txt"
    
    if image_file.exists() and mask_file.exists() and det_file.exists() and label_file.exists():
        continue
    else:
        print(file, 'bad')
        count += 1
        if image_file.exists(): image_file.unlink()
        if mask_file.exists(): mask_file.unlink()
        if det_file.exists(): det_file.unlink()
        if label_file.exists(): label_file.unlink()
        
print(f"{count} files are bad!")