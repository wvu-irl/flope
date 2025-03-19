import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from torchvision.io import read_image

from sunflower.utils.conversion import qvec2rotmat


class PoseNetFlowerDataset(Dataset):
    def __init__(self, path: str, test=False):
        dataset_path = Path(path)
        self.img_path = dataset_path/'images'
        self.pose_path = dataset_path/'poses'
        self.intrin_path = dataset_path/'intrinsics'
        self.all_files = os.listdir(self.img_path)
        self.all_files.sort()
        # self.all_files = self.all_files[:10]*100 # Overfitting Experiment
        n_files = len(self.all_files)
        split = int(0.8*n_files)
      
        #! Train-Test split 
        if test:
            self.files = self.all_files[split:]
        else:
            self.files = self.all_files[:split]
            

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        
        #! Image
        img = read_image(self.img_path/name).to(torch.float32)
        img /= 255.0
        
        #! Intirn
        intrin = np.loadtxt(self.intrin_path/f"{name[:-3]}txt", delimiter='\t')
        intrin = torch.as_tensor(intrin, dtype=torch.float32)

        #! Pose
        qt = np.loadtxt(self.pose_path/f"{name[:-3]}txt", delimiter='\t')
        q, t = qt[:4], qt[4:]
        rot = qvec2rotmat(q)
        
        rot = torch.as_tensor(rot, dtype=torch.float32)
        t = torch.as_tensor(t, dtype=torch.float32)
        
        return img, intrin, rot, t


if __name__=='__main__':
    np.set_printoptions(suppress=True, precision=3)
    torch.set_printoptions(sci_mode=False, precision=3)
    
    dataset = PoseNetFlowerDataset('/home/rashik_shrestha/data/flower_posenet_data')
    print(len(dataset))
    img, intrin, rot, trans = dataset[0]
    print(img.shape, intrin.shape, rot.shape, trans.shape)