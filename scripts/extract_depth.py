"""
Simple script to extract depth values of all the detected flowers and store
them in the depth_val dir
"""
import os
import numpy as np
import cv2
from tqdm import tqdm

from sunflower.utils.io import DatasetPath, pth
from sunflower.utils.image_manipulation import get_depth_value

if __name__=='__main__':
    data = DatasetPath('/home/rashik_shrestha/data/andy_data/single_cluster')
    depth_type = 'png'
    near_plane = 0.1
    far_plane = 3.0 # 2.0 for flowerur
    
    #!IO
    data.depth_val.mkdir(exist_ok=True, parents=True)
    
    for i, file in enumerate(tqdm(data.files)):
        det = np.loadtxt(pth(data.det, file, 'txt'))
        
        if det.shape[0] == 0: # If no detection
            np.savetxt(pth(data.depth_val, file, 'txt'), np.array([]))
            continue
        
        if len(det.shape) == 1: # If single detection
            det = det[None]
            
        bb = det[:,:4].astype(np.int16)
        
        #! select either one of below
        if depth_type=='png':
            depth = cv2.imread(pth(data.depth, file, 'png'), cv2.IMREAD_UNCHANGED)/1000
        elif depth_type=='npy':
            depth = np.load(pth(data.depth, file, 'npy' ))
        else:
            print('Depth format not reconized. Exiting.')
            exit()
        
        seg_mask = cv2.imread(pth(data.mask, file, 'png'), cv2.IMREAD_UNCHANGED)
        
        # print(file) 
        # print(bb.shape)
        # print(depth.shape, depth.dtype, depth.min(), depth.max())
        # print(seg_mask.shape, seg_mask.dtype, seg_mask.min(), seg_mask.max())
        
        depth_val, depth_reliable, _ = get_depth_value(
            bb, depth, seg_mask,
             near_plane = near_plane, far_plane = far_plane
        )
        
        data_to_log = np.vstack((depth_val, depth_reliable))
        
        np.savetxt(pth(data.depth_val, file, 'txt'), data_to_log)
        
    print("Done!")