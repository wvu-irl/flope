import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from torchvision.io import read_image
import cv2
from scipy.spatial.distance import cdist
import json
from icecream import ic
# ic.disable()

from sunflower.utils.conversion import rotmat2qvec
from sunflower.utils.io import DatasetPath, pth, load_pose, read_intrinsics_yaml_to_K_h_w
from sunflower.utils.mvg import homography_transform, get_crop_intrinsics, get_points3d, rotate_image, make_homogeneous
from sunflower.utils.mvg import nullify_yaw_batch
from sunflower.utils.geometry import sample_rectangle
from sunflower.utils.plot import plot_axis
from sunflower.utils.plot import plot_flower_poses_on_image

def draw_rectangle(img, corners, color=(0,255,0)):
    for i in range(len(corners)):
        start_point = np.array(corners[i])
        end_point = np.array(corners[(i + 1) % len(corners)])  # Connect back to the first corner
        cv2.line(img, start_point.astype(np.int16), end_point.astype(np.int16), color, 20)


class FlowerAttnDataset(Dataset):
    def __init__(self, path: str):
        self.data = DatasetPath(path)
        
        #! Read splat scale to scale the depth values
        with open(self.data.dataset/'dataparser_transforms.json', 'r') as f:
            splat_tf = json.load(f)
            splat_scale = splat_tf['scale']
            self.splat_scale = 1/splat_scale

 
    def __len__(self):
        return len(self.data.files)

    
    def bound_corners(self, corners, h, w):
        bounded_corners = []
        for x,y in corners:
            if x < 0: newx = 0
            elif x>w: newx=w
            else: newx=x
            if y<0: newy=0
            elif y>h: newy=h
            else: newy=y
            bounded_corners.append([newx, newy])
        return np.array(bounded_corners)
    

    def __getitem__(self, idx):
        name = self.data.files[idx]
        
        #! Read data
        img = cv2.imread(pth(self.data.rgb, name, 'jpg'))
        mask = cv2.imread(pth(self.data.mask, name, 'png'), cv2.IMREAD_UNCHANGED)
        det = np.loadtxt(pth(self.data.det, name, 'txt'))
        depth = np.load(pth(self.data.depth, name, 'npy'))*self.splat_scale
        depth_val, depth_reliable = np.loadtxt(pth(self.data.depth_val, name, 'txt')) #TODO: depth reliable not used
        depth_val *= self.splat_scale
        pose = load_pose(pth(self.data.pose, name, 'txt'))
        K,h,w = read_intrinsics_yaml_to_K_h_w(self.data.intrin)
        gt_poses = np.load(self.data.gt_poses)
      
        
        #! Get 3d flowers from this view (not the manually tuned GT flowers)
        depth_val = np.where(depth_val==0, 1e-3, depth_val) # handle depth=0
        
        uv = det[:,4:6]
        xyz = get_points3d(uv, depth_val, K)
        
        xyz_proj = (K@xyz.T).T
        xyz_proj /= xyz_proj[:,2].reshape(-1,1)
        xyz_proj = xyz_proj[:,:2]
       
        # for proj in xyz_proj: 
        #     cv2.circle(img, proj.astype(np.int16), radius=20, color=(0,0,255), thickness=-2)
        # cv2.imwrite('flower_detection.png', img)

        
        # world coordinate to camera coordinate 
        gt_poses_cam =  np.linalg.inv(pose)@gt_poses
        # gt_poses_cam =  pose@gt_poses
       
        # for gpc in gt_poses_cam: 
        #     plot_axis(img,gpc[:3,:3] , gpc[:3,3], K, thickness=5)
        
        # cv2.imwrite('projected.png', img)
        
        # print(gt_poses.shape, gt_poses_cam.shape)
        # print(pose.shape)
        # exit()
        
        
        # Rc,tc = pose[:3,:3], pose[:3,3]

        #! params
        #TODO update these parametes when using this dataset
        #TODO currently setup to generating metrics for teacher-student models evaluation
        # anchor_scale = 120 # for 1920x1080 image
        anchor_scale = 80 # for 1280x720 image (optimal for yolo model)
        size_var = 0  # how much to vary the size by
        
        out_h = 9
        out_w = 16
        out_scale = random.randint(anchor_scale-size_var, anchor_scale+size_var)

        
        scale = anchor_scale/out_scale
        
        out_h *= out_scale
        out_w *= out_scale
        
        #! Get random rotations
        Rx = random.uniform(-10, 10)
        Ry = random.uniform(-10, 10)
        Rz = random.uniform(-180, 180)
        random_rotation = (Rx,Ry,Rz)
      
        #! Apply Random Homography Transformation 
        img_rot, mask_rot, gt_rotations_cam_rot, gt_trans_cam_rot, transformed_corners = homography_transform(img, random_rotation, K, mask, gt_poses_cam[:,:3,:3], gt_poses_cam[:,:3,3]) 
        img_rot, rot_matrix, transformed_corners, homography = rotate_image(img, random_rotation, K) # reduntant here , used to get rot_matrix
        depth_rot, _, _, homography = rotate_image(depth, random_rotation, K)
        transformed_corners_bounded = self.bound_corners(transformed_corners, h, w)
        rect, shrinked_corners = sample_rectangle(transformed_corners_bounded, out_h, out_w)
       
        #! Crop and Scale: Image, Mask, Intrinsics 
        xmin, ymin = rect[0].astype(np.int16)
        xmax, ymax = rect[2].astype(np.int16)
        
        K_crop = get_crop_intrinsics(K, (xmin, ymin), scale) #! output

        img_crop = img_rot[ymin:ymax,xmin:xmax,:]
        
        if 0 in img_crop.shape:
            return None, None, None, None, None, None
       
        img_crop = cv2.resize(img_crop, (out_w,out_h)) #! output

        mask_crop = mask_rot[ymin:ymax,xmin:xmax]
        mask_crop = cv2.resize(mask_crop, (out_w,out_h)) #! output
        
        depth_crop = depth_rot[ymin:ymax,xmin:xmax]
        depth_crop = cv2.resize(depth_crop, (out_w,out_h)) #! output
        
        
        #! Filter flowers by visibility from this frame
        cam_transform = np.eye(4)
        cam_transform[:3,:3] = rot_matrix
       
        #! Project 
        xyz_rot = (cam_transform@make_homogeneous(xyz).T).T[:,:3]
        xyz_proj = (K@xyz_rot.T).T
        xyz_proj /= xyz_proj[:,2].reshape(-1,1)
        xyz_proj = xyz_proj[:,:2]
       
        # for proj in xyz_proj: 
        #     cv2.circle(img_rot, proj.astype(np.int16), radius=10, color=(0,0,255), thickness=-2)
            
        #! Project GT 
        gt_proj = (K@gt_trans_cam_rot.T).T
        gt_proj /= gt_proj[:,2].reshape(-1,1)
        gt_proj = gt_proj[:,:2]
       
        # for proj in gt_proj: 
        #     cv2.circle(img_rot, proj.astype(np.int16), radius=10, color=(0,255,0), thickness=-2)
            
        #! Mask to filter out points lying outside window of interest
        in_frame_mask = []
        for x,y in xyz_proj:
            if x>xmin and x<xmax and y>ymin and y<ymax:
                in_frame_mask.append(True)
            else:
                in_frame_mask.append(False)
        in_frame_mask = np.array(in_frame_mask)
            
        #! Compare GT flowers and flowers in this frame
        points_dist = cdist(xyz_proj, gt_proj, metric='euclidean')
        min_dist = np.min(points_dist, axis=1) 
        min_dist_idx = np.argmin(points_dist, axis=1) 
        vis_gt_flowers_mask = min_dist < 20

        #! Find the GT flowers closest to flowers in this frame         
        vis_gt_flowers_trans = gt_trans_cam_rot[min_dist_idx]
        vis_gt_flowers_rot = gt_rotations_cam_rot[min_dist_idx]

        #! Master mask to Filter out GT flowers that are either too far from 
        #! the flowers in this frame of outside the window of interest
        master_filter = np.logical_and(in_frame_mask, vis_gt_flowers_mask) 
       
        #! Filter out  
        vis_gt_flowers_trans = vis_gt_flowers_trans[master_filter] #! output
        vis_gt_flowers_rot = vis_gt_flowers_rot[master_filter] #! output
        # print(vis_gt_flowers_trans.shape, vis_gt_flowers_rot.shape)
        
        if vis_gt_flowers_rot.shape[0] == 0:
            return None, None, None, None, None, None
        
        #! Project Visible GT
        xyz_proj = (K@vis_gt_flowers_trans.T).T
        xyz_proj /= xyz_proj[:,2].reshape(-1,1)
        xyz_proj = xyz_proj[:,:2]
       
        # for proj in xyz_proj: 
        #     cv2.circle(img_rot, proj.astype(np.int16), radius=10, color=(255,0,0), thickness=-2)
        
        # cv2.imwrite('flower_detection_in_rotated.png', img_rot)
        # print('done')
        

        # gt_poses_cam_transform =  cam_transform@gt_poses_cam
        # cam_transform = np.eye(4)
        # cam_transform[:3,:3] = rot_matrix

        # gt_poses_cam_transform =  cam_transform@gt_poses_cam
       
        # print('this is rotmat') 
        # print(rot_matrix)
        # print(cam_transform)
        # print(gt_poses_cam_transform)
        # print(rect)
        
               
        # for gpc in gt_poses_cam_transform: 
        #     plot_axis(img_rot, gpc[:3,:3] , gpc[:3,3], K, thickness=5)
            
        
        # draw_rectangle(img_rot, transformed_corners_bounded)
        # draw_rectangle(img_rot, shrinked_corners)
        # draw_rectangle(img_rot, rect, (255,0,0))
        
      
        # print(Rc,tc) 
        # print(img.shape, pose.shape)
        
       

        # cv2.imwrite('rotated.png', img_rot)
        
        vis_gt_flowers_rot = nullify_yaw_batch(vis_gt_flowers_rot) 
       
        # Combine R and t to get Rt matrix
        final_flower_poses = np.repeat(np.eye(4)[None], vis_gt_flowers_rot.shape[0], axis=0)
        final_flower_poses[:,:3,:3] = vis_gt_flowers_rot
        final_flower_poses[:,:3,3] = vis_gt_flowers_trans
        
        # print(vis_gt_flowers_rot[0])
        # print(vis_gt_flowers_trans[0])
        # print(final_flower_poses[0])
       
        #! Plot the poses on the image to see if data is good 
        # plot_flower_poses_on_image(img_crop, final_flower_poses, K_crop)
        # cv2.imwrite('image_with_plots.png', img_crop)
        
        #! Convert to Torch
        img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
        img_tensor = torch.as_tensor(img_crop)/255
        img_tensor = img_tensor.permute(2,0,1)
        # print(img_tensor.shape, img_tensor.dtype, img_tensor.min(), img_tensor.max())
        
        mask_tensor = torch.as_tensor(mask_crop)>128
        # print(mask_tensor.shape, mask_tensor.dtype, mask_tensor.min(), mask_tensor.max())
        
        depth_tensor = torch.as_tensor(depth_crop)
        # print(depth_tensor.shape, depth_tensor.dtype, depth_tensor.min(), depth_tensor.max())
         
        pose_tensor = torch.as_tensor(final_flower_poses) 
        
        K_tensor = torch.as_tensor(K_crop)
        
        #! Make the final number of flower poses exactly 15, either pad, or remove
        num_of_flowers_detected = pose_tensor.shape[0]
        print(f"Initial number of flowers")
        if num_of_flowers_detected > 15:
            pose_tensor = pose_tensor[:15]
        elif num_of_flowers_detected < 15:
            pad = 15-num_of_flowers_detected
            identity_poses = torch.as_tensor(np.repeat(np.eye(4)[None], pad, axis=0))
            pose_tensor = torch.concat((pose_tensor, identity_poses), dim=0)
            
        #! Returns
        return img_tensor, mask_tensor, depth_tensor, pose_tensor, K_tensor, num_of_flowers_detected


if __name__=='__main__':
    from tqdm import tqdm
    
    dataset = FlowerAttnDataset('/home/rashik_shrestha/data/sunflower/plantscan_pixel_1230_raw')
    epochs = 1
    ic(len(dataset))
 
    #! Plot flowers detected histogram
    # all_count = []
    # for _ in range(epochs): 
    #     for i in tqdm(range(len(dataset))):
    #         img, mask, depth, poses, K, N = dataset[i]
    #         if img is None:
    #             continue
    #         ic(img.shape, mask.shape, depth.shape, poses.shape, K.shape, N)
    #         all_count.append(poses.shape[0])
            
    # all_count = np.array(all_count)
    
    # import matplotlib.pyplot as plt
    # plt.hist(all_count)
    # plt.savefig('count_hist.png')

    #! Plot single datapoint
    # img, mask, depth, poses, K, N = dataset[0] 
    # # apply depth colormap
    # depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    # depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    
    # # blend all three images for visualization
    # mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # blended = 0.4*img + 0.3*depth_colormap + 0.3*mask_color 
    
    # cv2.imwrite('image.png', img)
    # cv2.imwrite('mask.png', mask)
    # cv2.imwrite('depth.png', depth_colormap)
    # cv2.imwrite('blended.png', blended)