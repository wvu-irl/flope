import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from torchvision.io import read_image

from sunflower.utils.conversion import rotmat2qvec

class FlowerDataset(Dataset):
    def __init__(self, path: str):
        dataset_path = Path(path)
        self.img_path = dataset_path/'color'
        self.pose_path = dataset_path/'poses_ba'
        self.files = os.listdir(self.img_path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = read_image(self.img_path/name).to(torch.float32)
        img /= 255.0
        pose = np.loadtxt(self.pose_path/f"{name[:-3]}txt")
        R = pose[:3,:3]
        q = rotmat2qvec(R)
        q = torch.as_tensor(q, dtype=torch.float32)
        return img, q


if __name__=='__main__':
    dataset = FlowerDataset('/home/rashik_shrestha/ws/OnePose/data/onepose_datasets/sample_data/flowerprop/flowerprop-3')
    print(len(dataset))
    dataset[0]