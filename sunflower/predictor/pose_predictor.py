"""

Input: RGB, Depth, CameraPose
Output: Flower Poses

"""

import os
from tqdm import tqdm
from pathlib import Path
import torch
from sunflower.dataset.posenet_flower_dataset import PoseNetFlowerDataset
from sunflower.models.posenet import PoseResNet
import roma
import cv2
import numpy as np
from PIL import Image
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from numpy import ndarray
from icecream import ic
np.set_printoptions(precision=3, suppress=True)

#! Sunflower
from sunflower.utils.image_manipulation import change_contrast
from sunflower.utils.plot import plot_axis_and_translate, plot_bounding_boxes, plot_flower_poses_on_image
from sunflower.utils.conversion import procrustes_to_rotmat
from sunflower.models.grounding_dino import GroundingDINO
from sunflower.models.sam import SAM
from sunflower.utils.data import get_pixel6a_cam_matrix, get_realsense_435_cam_matrix
from sunflower.utils.mvg import squarify_bb, bb_in_frame, filter_very_large_bb, get_points3d
from sunflower.utils.io import read_intrinsics_yaml_to_K_h_w, DatasetPath, pth
from sunflower.utils.image_manipulation import get_depth_value
from sunflower.utils.mvg import nullify_yaw_batch

log = logging.getLogger(__name__)


class PosePredictor():
    def __init__(self,
        device: str,
        posenet_path: str,
        intrin_path: str,
        debug: bool = False
    ):
        self.device = device
        self.debug = debug
    
        #! Posenet
        self.posenet = PoseResNet().to(device)
        self.posenet.load_state_dict(torch.load(posenet_path, weights_only=True))
        log.info(f"Model loaded: {Path(posenet_path).name}")

        #! Grounding Dino (Flowers Detection)
        self.gdino = GroundingDINO(device, 'white flower.', box_th=0.3, text_th=0.3, obj_filter='white flower')
        log.info(f"Grounding DINO loaded")

        #! SAM (Mask Generator)
        self.sam = SAM(device)
        log.info(f"SAM loaded")

        #! Intrinsics
        self.K, self.height, self.width = read_intrinsics_yaml_to_K_h_w(intrin_path)
        
        log.info("PosePredictor initialized!")
        

    def get_flower_poses(
        self,
        rgb: ndarray,
        depth: ndarray,
    ):
        #! Flower Detection
        bb_dino = self.gdino.detect(rgb)
        if bb_dino.shape[0] == 0:
            ic('No bbox detected by dino!')
            return None
        ic(f"{bb_dino.shape} bbox detected by dino.")

        # if bb_dino.shape[0] < 3: #TODO: Ignore less than 3 flowers
        #     return None
        bb_dino = filter_very_large_bb(bb_dino)
        ic(f"{bb_dino.shape} bbox remains after filtering very large bbox.")

        #! Segmentation
        rgb_pil = Image.fromarray(rgb)
        mask = self.sam.get_segmentation_mask(rgb_pil, bb_dino.tolist()) #TODO: redundancy here, seg mask first joint, then cropped again
        # cv2.imwrite('r405_mask.png', mask)

        #! Squarify bb and filter 
        sq_bb = []
        uv = []
        good_bb = []
        for bb in bb_dino:
            xmin, ymin, xmax, ymax = bb
            # u = xmin+(xmax-xmin)/2 - self.width/2
            # v = ymin+(ymax-ymin)/2 - self.height/2
            u = (xmax+xmin)/2
            v = (ymax+ymin)/2
            sbb = squarify_bb(bb)
            if not bb_in_frame(sbb, rgb.shape):
                continue
            uv.append([u,v])
            sq_bb.append(sbb)
            good_bb.append(bb)
        uv = np.array(uv)
        good_bb = np.array(good_bb).astype(np.int16)
        sq_bb = np.array(sq_bb)
        
        ic(f"{good_bb.shape} bbox remains after filtering out bbox outside the frame.")
       
        # Return None if no good bb 
        if good_bb.shape[0] == 0:
            return None

        #! Get Depth Values
        depth = depth.astype(np.float32)/10000 # converting depth to meters for realsense 405
        depth_val, depth_reliable, _ = get_depth_value(
            good_bb, depth, mask,
             near_plane = 0.1, far_plane = 2.5
        )

        #! Filter out unreliable depth values
        depth_val = depth_val[depth_reliable]
        uv = uv[depth_reliable]
        sq_bb = sq_bb[depth_reliable]
        
        ic(f"{sq_bb.shape} flowers remain after filtering out bad depth.")
        
        if sq_bb.shape[0] == 0:
            return None

        #! Lift 2d points to 3d using depth
        xyz = get_points3d(uv, depth_val, self.K)

        #! Create batch of flower crops
        image_batch_np = [] 
        for bb in sq_bb: 
            xmin, ymin, xmax, ymax = bb
            
            img_crop = rgb[ymin:ymax, xmin:xmax]
            mask_crop = mask[ymin:ymax, xmin:xmax]
                    
            img_crop_sized = cv2.resize(img_crop, (512,512), interpolation=cv2.INTER_LANCZOS4)
            mask_crop_sized = cv2.resize(mask_crop, (512,512), interpolation=cv2.INTER_LANCZOS4)
            
            img_crop_sized_nobg = img_crop_sized * (mask_crop_sized.reshape(512,512,1)/255.0)
            image_batch_np.append(img_crop_sized_nobg)
            
        image_batch_np = np.array(image_batch_np)/255.0
        image_batch = torch.as_tensor(image_batch_np, dtype=torch.float32)
        image_batch = torch.permute(image_batch, (0,3,1,2)).to(self.device)

        #! Use PoseNet to get Rotation Matrices
        r9_M_pred = self.posenet(image_batch) 
        rot_pred = procrustes_to_rotmat(r9_M_pred)
        rot_pred_np = rot_pred.detach().cpu().numpy()  # (B,3,3)
        
        #! Nullify Yaw
        rot_pred_np = nullify_yaw_batch(rot_pred_np)

        #! Plot axis quick
        # img_clone = rgb.copy()
        # t = np.array([0,0,1])
        # for R, bb in zip(rot_pred_np, good_bb):
        #     plot_axis_and_translate(img_clone, R, t, self.K, bb, self.height, self.width, 15)
        # cv2.imwrite('/home/rashik_shrestha/ws/sunflower/output/ros_callback/quick.png', img_clone) 


        #! Combile rotations and translation
        Rt = np.repeat(np.eye(4)[None], rot_pred_np.shape[0], axis=0)
        Rt[:,:3,:3] = rot_pred_np
        Rt[:,:3,3] = xyz

        #! Rotate and translate as per ROS2 convention and end effector length
        # roter_translator = np.array([
        #     [  1.0000000,  0.0000000,  0.0000000, 0.01],
        #     [0.0000000,  -1.0000000,  0.0000000, 0],
        #     [0.0000000,  0.0000000,  -1.0000000, 0.28 ],
        #     [0,0,0,1]
        # ])
        # Rt = Rt@roter_translator

        #! Return Flower Rotations
        return Rt

 
if __name__ == '__main__':
    import cv2
    import numpy as np

    #! Inputs paths
    # rgb_path = '/home/rashik_shrestha/data/sunflower/flowerur/rgb/frame_00000.png'
    rgb_path = '/home/rashik_shrestha/data/flower_lib/rgb/frame_00010.png'
    # depth_path = '/home/rashik_shrestha/data/sunflower/flowerur/depth/frame_00000.png'
    depth_path = '/home/rashik_shrestha/data/flower_lib/depth/frame_00010.png'
    cam_pose_path = '/home/rashik_shrestha/data/sunflower/flowerur/pose/frame_00000.txt'
    intirin_path = '/home/rashik_shrestha/data/sunflower/flowerur/intrinsics.yaml'

    #! Read inputs
    rgb = cv2.imread(rgb_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    cam_pose = np.loadtxt(cam_pose_path)

    model = PosePredictor(
        device='cuda',
        posenet_path='/home/rashik_shrestha/ws/sunflower/scripts/weights/posenet_e183.pth',
        intrin_path=intirin_path,
        debug=True
    )

    K,h,w = read_intrinsics_yaml_to_K_h_w(intirin_path)

    flower_pose = model.get_flower_poses(rgb, depth)

    plot_flower_poses_on_image(rgb, flower_pose, K)

    cv2.imwrite('axis_projected.png', rgb)

    print(flower_pose.shape)

