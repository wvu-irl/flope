import os
import yaml
import numpy as np
import plyfile
from pathlib import Path
from typing import List

from dataclasses import dataclass, field


@dataclass
class DatasetPath:
    path: str
    dataset: Path = field(init=False)
    rgb: Path = field(init=False)
    depth: Path = field(init=False)
    det: Path = field(init=False)
    mask: Path = field(init=False)
    pose: Path = field(init=False)
    splats: Path = field(init=False)
    intrin: Path = field(init=False)
    depth_val: Path = field(init=False)
    files: List[str] = field(init=False)
    aligned: Path = field(init=False)
    gt_poses: Path = field(init=False)
    
    vis_det: Path = field(init=False) # Detect Flower Bounding boxes
    vis_pose: Path = field(init=False) # Predicted Flower poses
    vis_gt: Path = field(init=False) # Gt flower poses projected to images.
    
    def __post_init__(self):
        self.dataset = Path(self.path)
        self.rgb = self.dataset/'rgb'
        self.depth = self.dataset/'depth'
        self.det = self.dataset/'detection'
        self.mask = self.dataset/'mask'
        self.pose = self.dataset/'pose'
        self.splats = self.dataset/'splats.ply'
        self.intrin = self.dataset/'intrinsics.yaml'
        self.depth_val = self.dataset/'depth_val'
        self.vis_det = self.dataset/'vis_det'
        self.vis_pose = self.dataset/'vis_pose'
        self.vis_gt = self.dataset/'vis_gt'
        self.aligned = self.dataset/'aligned'
        self.gt_poses = self.dataset/'gt_poses.npy'

        #! Read files list       
        rgb_files = os.listdir(self.rgb) 
        rgb_files.sort()
        
        #! Read bad files
        with open(self.dataset/'bad.txt', 'r') as fp:
            bad_files = [line.strip() for line in fp.readlines()]
           
        #! Prepare files list excluding bad ones 
        self.files = []
        for rfile in rgb_files:
            fname = rfile[:-4]
            if fname not in bad_files:
                self.files.append(fname)


def get_files(path):
    files = os.listdir(path)
    files.sort()
    files_no_ext = [f[:-4] for f in files]
    return files_no_ext

def load_pose(path):
    """Load camera pose R9,t3
    
    Returns:
        Pose 4x4
    """
    pose = np.loadtxt(path)
    rotmat, trans = pose[:9].reshape(3,3), pose[9:]
    pose_4by4 = np.eye(4)
    pose_4by4[:3,:3] = rotmat
    pose_4by4[:3,3] = trans
    return pose_4by4
    
 
def pth(path: Path, name: str, ext: str):
    return path/f"{name}.{ext}"


def read_intrinsics_yaml(filepath: str):
    with open(filepath, "r") as yaml_file:
        data = yaml.safe_load(yaml_file)
    return data

def read_intrinsics_yaml_to_K_h_w(filepath: str):
    data = read_intrinsics_yaml(filepath)
    return np.array([
        [data['fx'], 0, data['cx']],
        [0, data['fy'], data['cy']],
        [0,0,1]
    ]), data['h'], data['w']
    
    
def render_video(path, fps=10):
    cmd = f"/usr/bin/ffmpeg -framerate {fps} -i {path}/frame_%05d.png -y -c:v libx264 -pix_fmt yuv420p {path}/all_frames.mp4"
    print("Running command:", cmd)
    os.system(cmd)
   

def read_splats_ply(splats_path):
    with open(splats_path, "rb") as f:
        ply_data = plyfile.PlyData.read(f)
    
    vertex_data = ply_data["vertex"]

    x = np.array(vertex_data["x"])
    y = np.array(vertex_data["y"])
    z = np.array(vertex_data["z"])

    r = np.array(vertex_data["f_dc_0"])
    g = np.array(vertex_data["f_dc_1"])
    b = np.array(vertex_data["f_dc_2"])

    points = np.array([x,y,z]).T
    colors = np.array([r,g,b]).T

    colors = (colors-colors.min())/(colors.max()-colors.min())
    
    return points, colors