import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random

if __name__=='__main__':
    """
    Adds random agumentation to images
    """
    #! Argument parser
    parser = argparse.ArgumentParser(description="Add random image agumentations")
    parser.add_argument("-i", "--input", type=str, help="Path to the input directory", required=True)
    parser.add_argument("-o", "--output", type=str, help="Path to the output directory", required=True)
    parser.add_argument("-r", "--random-seed", type=int, help="Random seed", default=0)
    args = parser.parse_args()

    #! Random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    #! Paths
    input_dir = Path(args.input) 
    output_dir = Path(args.output) 
    if not input_dir.exists(): print(f"Input dir {input_dir} doesn't exist. Exiting")
    output_dir.mkdir(exist_ok=True, parents=True)

    #! Define transformation
    transform = transforms.Compose([
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3),  # Random perspective with 50% probability
        transforms.RandomRotation(degrees=180),                                       # Random rotation within ±180 degrees
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1), # Random color jitter
        transforms.RandomGrayscale(p=0.2)                                            # Random grayscaling with 20% probability
    ])

    #! Get Images
    images = os.listdir(input_dir)
    # images = images*5 # generate 5 different varients of same datapoint
    print(f"Generation {len(images)} datapoints.")

    count = 0
    for img_name in tqdm(images):
        image = Image.open(input_dir/img_name)
        trans = transform(image)
        trans.save(output_dir/f"{count:07d}.png")
        count += 1