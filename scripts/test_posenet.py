"""
Given set of images, get flower poses of every detected flower on each image.
Optionally draw bounding box of detected flower.
Optionally draw estimated flower poses as well.
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
np.set_printoptions(precision=3, suppress=True)

#! Sunflower
from sunflower.utils.image_manipulation import change_contrast
from sunflower.utils.plot import plot_axis_and_translate, plot_bounding_boxes
from sunflower.utils.conversion import procrustes_to_rotmat
from sunflower.models.grounding_dino import GroundingDINO
from sunflower.models.sam import SAM
from sunflower.utils.data import get_pixel6a_cam_matrix, get_realsense_435_cam_matrix
from sunflower.utils.mvg import squarify_bb, bb_in_frame, filter_very_large_bb
from sunflower.utils.io import read_intrinsics_yaml_to_K_h_w, DatasetPath, pth

log = logging.getLogger(__name__)

    
@hydra.main(version_base=None, config_path="./config", config_name="test_posenet")
def main(cfg: DictConfig):
    log.info("Config:")
    log.info(OmegaConf.to_yaml(cfg))
    
    #! I/O
    data = DatasetPath(Path(cfg.data_dir))
    
    #make dirs if not available
    data.vis_det.mkdir(exist_ok=True, parents=True)
    data.mask.mkdir(exist_ok=True, parents=True)
    data.det.mkdir(exist_ok=True, parents=True)
    data.vis_pose.mkdir(exist_ok=True, parents=True)

    #! Model
    model = PoseResNet().to(cfg.device)
    model.load_state_dict(torch.load(cfg.weights, weights_only=True))
    log.info(f"Model loaded: {Path(cfg.weights).name}")
    gdino = GroundingDINO(cfg.device, 'white flower.', box_th=0.3, text_th=0.3, obj_filter='white flower')
    log.info(f"Grounding DINO loaded")
    sam = SAM(cfg.device)
    log.info(f"SAM loaded")
    
    #! Get Intrinsics
    K, h, w = read_intrinsics_yaml_to_K_h_w(data.intrin)
   
    #! Get Image files 
    img_files = os.listdir(data.rgb)
    img_files.sort()
    
    #! For each image:
    for img_file in tqdm(img_files):
        #! Out filename
        out_name = img_file.rsplit('.',1)[0]
       
        #! Get Input Image 
        img_pil = Image.open(pth(data.rgb,out_name,'png')) # PIL (RGB)
        img_high_contrast = change_contrast(img_pil) # OpenCV BGR
        img_cv = np.array(img_pil) # OpenCV RGB
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR) # OpenCV BGR
        
        #! Detection
        bb_dino = gdino.detect(img_cv)
        
        if bb_dino.shape[0] == 0:
            cv2.imwrite(pth(data.vis_det,out_name,"png"), img_cv) # Bounding box vis null
            cv2.imwrite(pth(data.mask, out_name, 'png'), np.zeros_like(img_cv)) # Mask null
            np.savetxt(pth(data.det, out_name, 'txt'), np.array([]), fmt='%.7f') # detection null
            cv2.imwrite(pth(data.vis_pose, out_name, 'png'), img_cv)  # pose vis null
            continue
            
        bb_dino = filter_very_large_bb(bb_dino)
        
        if bb_dino.shape[0] == 0:
            cv2.imwrite(pth(data.vis_det,out_name,"png"), img_cv) # Bounding box vis null
            cv2.imwrite(pth(data.mask, out_name, 'png'), np.zeros_like(img_cv)) # Mask null
            np.savetxt(pth(data.det, out_name, 'txt'), np.array([]), fmt='%.7f') # detection null
            cv2.imwrite(pth(data.vis_pose, out_name, 'png'), img_cv)  # pose vis null
            continue
        
        img_det = plot_bounding_boxes(img_cv.copy(), bb_dino)
    
        # print(f"{len(bb_dino)} flowers detected.")
        # img_det_small = cv2.resize(img_det, (403, 302))     
        cv2.imwrite(pth(data.vis_det,out_name,"png"), img_det)
        
        #! Get Seg mask
        mask = sam.get_segmentation_mask(img_pil, bb_dino.tolist())
        cv2.imwrite(pth(data.mask, out_name, 'png'), mask)
        
        #! Get square bbs
        filter_bb = []
        sq_bb = []
        for bb in bb_dino:
            sbb = squarify_bb(bb)
            if not bb_in_frame(sbb, img_cv.shape):
                continue
            sq_bb.append(sbb)
            filter_bb.append(bb)
            
        filter_bb = np.array(filter_bb)
        
        if filter_bb.shape[0] == 0:
            cv2.imwrite(pth(data.vis_det,out_name,"png"), img_cv) # Bounding box vis null
            cv2.imwrite(pth(data.mask, out_name, 'png'), np.zeros_like(img_cv)) # Mask null
            np.savetxt(pth(data.det, out_name, 'txt'), np.array([]), fmt='%.7f') # detection null
            cv2.imwrite(pth(data.vis_pose, out_name, 'png'), img_cv)  # pose vis null
            continue
        
        image_batch_np = [] 
        for bb in sq_bb: 
            xmin, ymin, xmax, ymax = bb
            
            img_crop = img_cv[ymin:ymax, xmin:xmax]
            mask_crop = mask[ymin:ymax, xmin:xmax]
                    
            img_crop_sized = cv2.resize(img_crop, (512,512), interpolation=cv2.INTER_LANCZOS4)
            mask_crop_sized = cv2.resize(mask_crop, (512,512), interpolation=cv2.INTER_LANCZOS4)
            
            img_crop_sized_nobg = img_crop_sized * (mask_crop_sized.reshape(512,512,1)/255.0)
            # print(img_crop_sized_nobg.shape)
            image_batch_np.append(img_crop_sized_nobg)
            
        image_batch_np = np.array(image_batch_np)/255.0
        image_batch = torch.as_tensor(image_batch_np, dtype=torch.float32)
        image_batch = torch.permute(image_batch, (0,3,1,2)).to(cfg.device)
        # print(image_batch_np.shape, image_batch_np.min(), image_batch_np.max())
        r9_M_pred = model(image_batch) 
        # print(r9_M_pred.shape)
        rot_pred = procrustes_to_rotmat(r9_M_pred)
        # print(rot_pred.shape)
        
        rot_pred_np = rot_pred.detach().cpu().numpy() 
    
        #! Write info 
        all_data = []
        for w_bb, w_rot in zip(filter_bb, rot_pred_np):
            xmin, ymin, xmax, ymax = w_bb
            x = (xmin+xmax)/2
            y = (ymin+ymax)/2
            rot_flat = w_rot.flatten().tolist()
            this_data = w_bb.tolist() + [x.item(),y.item()] + rot_flat
            all_data.append(this_data)
        all_data = np.array(all_data)
        
        # print(all_data.shape)
        np.savetxt(pth(data.det, out_name, 'txt'), all_data, fmt='%.7f')

    
        t = np.array([0,0,1])
        
        #! Plot axis
        for R, bb in zip(rot_pred_np, sq_bb):
            plot_axis_and_translate(img_cv, R, t, K, bb, h, w, 15)
    
        # img_cv = cv2.resize(img_cv, (403, 302))     
        cv2.imwrite(pth(data.vis_pose, out_name, 'png'), img_cv) 
        

if __name__=='__main__':
    main()