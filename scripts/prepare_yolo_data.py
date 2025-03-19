"""
This is to prepare data to train YoLo for flower detection and segmentation.
"""
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from sunflower.models.grounding_dino import GroundingDINO
from sunflower.models.sam import SAM
# from sunflower.utils.plot import plot_bounding_boxes
from sunflower.utils.mvg import filter_very_large_bb
from pathlib import Path
from icecream import ic
from tqdm import tqdm
from tyro import cli

# torch.cuda.set_device(1)
ic.disable()

def main(
    thread_id: int = 0,
    split: str = 'train'
):
    device = 'cuda'
    raw_data_dir = '/users/rs00140/dataset/yolo_seg_raw'
    out_dir = '/users/rs00140/dataset/yolo_seg'

    #! I/O
    raw_data_dir = Path(raw_data_dir)
    out_dir = Path(out_dir)
    files = raw_data_dir.rglob("*.png")
    files = list(files)
    files.sort()
    print(f"{len(files)} files available.")

    #! Models
    gdino = GroundingDINO(device, 'white flower.', box_th=0.3, text_th=0.3, obj_filter='white flower')
    ic("Grounding DINO loaded")
    sam = SAM(device)
    ic("SAM loaded")

    #! Define transforms
    spatial_transform_pixel = transforms.Compose([
        transforms.RandomPerspective(distortion_scale=0.2, p=0.2),  # Random perspective
        transforms.RandomRotation(degrees=30),                      # Random rotation (e.g., ±30 degrees)
        transforms.RandomHorizontalFlip(p=0.5),                     # Random horizontal flip
        transforms.RandomVerticalFlip(p=0.1),                       # Random vertical flip
        transforms.RandomCrop(size=(1920,1080))
    ])

    spatial_transform = transforms.Compose([
        transforms.RandomPerspective(distortion_scale=0.2, p=0.2),  # Random perspective
        transforms.RandomRotation(degrees=30),                      # Random rotation (e.g., ±30 degrees)
        transforms.RandomHorizontalFlip(p=0.5),                     # Random horizontal flip
        transforms.RandomVerticalFlip(p=0.1),                       # Random vertical flip
    ])

    photometric_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
    ])

    count = 0
    for file in tqdm(files):
        try:
            cam_type = str(file).split('/')[-2]
            ic(cam_type)

            image = Image.open(file)

            if cam_type=='plantscan_pixel_1230_raw':
                transformed_image = spatial_transform_pixel(image)
            else:
                transformed_image = spatial_transform(image)

            transformed_image_np = np.array(transformed_image)
            transformed_image_np = cv2.cvtColor(transformed_image_np, cv2.COLOR_RGB2BGR)
            transformed_image_jittered = photometric_transform(transformed_image)

            #! Detection
            bb_dino = gdino.detect(transformed_image_np)
            if bb_dino.shape[0] < 3:
                continue
            
            bb_dino = filter_very_large_bb(bb_dino)
            # bb_plot = plot_bounding_boxes(transformed_image_np, bb_dino)
            # reject less than 3 detections
            if bb_dino.shape[0] < 3:
                continue
            ic(bb_dino.shape)

            #! Segmentation
            mask = sam.get_segmentation_mask(transformed_image, bb_dino.tolist())

            #! Save Images
            cv2.imwrite(out_dir/'masks'/split/f"frame{thread_id}_{count:07d}.png", mask)
            transformed_image_jittered.save(out_dir/'images'/split/f"frame{thread_id}_{count:07d}.png")
            np.savetxt(out_dir/'dets'/split/f"frame{thread_id}_{count:07d}.txt", bb_dino)
            # cv2.imwrite('bb_plot.png', bb_plot)
            # transformed_image.save('haha.png')

            #! Increment the count
            count += 1
        except:
            print(f"Error with: {file}")


    print("Data generation completed :)")


if __name__=='__main__':
    cli(main)