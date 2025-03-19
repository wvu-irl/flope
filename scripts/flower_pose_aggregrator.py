import os
import random
from pathlib import Path
import numpy as np
import cv2
import pickle
from tqdm import tqdm
from scipy.spatial.distance import cdist
import pickle
from scipy.spatial.transform import Slerp, Rotation as R

#! sunflower imports
from sunflower.utils.mvg import rot_average, get_points3d, pose_cam_to_world
from sunflower.utils.image_manipulation import get_depth_value
from sunflower.utils.conversion import get_pose_mat, qvec2rotmat, rotmat2qvec
from sunflower.utils.plot import plot_3D_poses, generate_rainbow_colors
from sunflower.utils.io import read_intrinsics_yaml_to_K_h_w


np.set_printoptions(suppress=True, precision=3)


class Env3D():
    def __init__(self, th=40, score_th=200):
        """
        Args:
            th (float): Threshold between two flowers to consider them different, in mm
        """
        self.th = th/1000
        self.score_th = score_th
        self.num = 0 # not used now
        self.trans = None # (self.num,3)
        self.quat = None # (self.num,4)
        self.score = None # (self.num,)
        
        self.all_new_trans = []
        self.all_new_quat = []
    
    def add_measurement(self, tvec, qvec):
        """
        Args:
            tvec: (N, 3)
            qvec: (N, 4)
        """
        #! First Measurement
        if self.trans is None:
            self.trans = tvec
            self.quat = qvec
            self.score = np.ones(tvec.shape[0])
           
            #! Log first newly added measurement 
            self.all_new_trans.append(tvec)
            self.all_new_quat.append(qvec)
            
            return
        
        #! Calculate distances 
        distance_matrix = cdist(tvec, self.trans, metric='euclidean')
        min_idx = np.argmin(distance_matrix, axis=1)
        min_vals = np.min(distance_matrix, axis=1)
        
        #! Filter good matches
        good_match = min_vals < self.th
        min_idx_good = min_idx[good_match]
        tvec_good = tvec[good_match]
        qvec_good = qvec[good_match]
        
        #! Append State
        state_score = self.score[min_idx_good]
        meas_score = np.ones(state_score.shape[0])
        normalizer = state_score + meas_score
        state_weight = state_score/normalizer
        meas_weight = meas_score/normalizer
        
        if min_idx_good.shape[0] == 0: #! Warning: None of the new flower positions matched with old one!
            self.trans = np.vstack((self.trans, tvec))
            self.quat = np.vstack((self.quat, qvec))
            self.score = np.concatenate((self.score, np.ones(tvec.shape[0])))
            
        else: 
            # print(f"{min_idx_good.shape[0]} Flowers matches to previous")
            #! Update Trans state
            self.trans[min_idx_good] = self.trans[min_idx_good]*state_weight.reshape(-1,1) + tvec_good*meas_weight.reshape(-1,1)
            
            #! Update Quat state
            self.quat[min_idx_good] = rot_average(self.quat[min_idx_good], qvec_good, state_weight, meas_weight)
            
            #! Log the newly added measurements
            new_trans = np.zeros_like(self.trans)
            new_trans[min_idx_good] = tvec_good
            self.all_new_trans.append(new_trans)

            new_quat = np.zeros_like(self.quat) # use (0,0,0,1) here instead
            new_quat[min_idx_good] = qvec_good
            self.all_new_quat.append(new_quat)

            #! Update score
            self.score[min_idx_good] += 1 
            
            xyz_unmatches = tvec[np.logical_not(good_match)]
            quat_unmatches = qvec[np.logical_not(good_match)]
            self.trans = np.vstack((self.trans, xyz_unmatches))
            self.quat = np.vstack((self.quat, quat_unmatches))
            self.score = np.concatenate((self.score, np.ones(xyz_unmatches.shape[0])))
    
         
    def save_filtered_data(self):
        print(f"Total unique flowers: {self.trans.shape[0]}")
        print(f"Max score: {self.score.max()}")
        print(f"Min score: {self.score.min()}")
        data = {
            'trans': self.trans,
            'quat': self.quat,
            'score': self.score
        }
        with open('filtered_data.pkl', 'wb') as fp:
            pickle.dump(data, fp)
        print("Filtered data saved to filtered_data.pkl")
    
        
    def get_final_data(self):
        score_filter = self.score>self.score_th
        return self.trans[score_filter], self.quat[score_filter]

 
    def save_measurements(self):
        with open('meas.pkl', 'wb') as fp:
            all_data = {
                'trans': self.all_new_trans,
                'quat': self.all_new_quat
            }
            pickle.dump(all_data, fp)
        print("All measurements saved to meas.pkl file")
        

if __name__=='__main__':
    import json
    
    #! Configs
    data_path = '/home/rashik_shrestha/data/sunflower/plantscan_pixel_1230_raw'
    dist_th = 10
    score_th = 100
    depth_near_th = 0.1
    depth_far_th = 2.5

    #! Random seed
    random.seed(0)
    np.random.seed(0)
    
    #! I/O
    data_dir = Path(data_path)
    K, h, w = read_intrinsics_yaml_to_K_h_w(data_dir/'intrinsics.yaml')
    files = os.listdir(data_dir/'rgb')
    files.sort()
    files = files[5:]
    nfiles = len(files)
    print(f"{nfiles} files available.")
    # splat transform
    splat_transforms_pth = '/home/rashik_shrestha/outputs/plantscan_pixel_1230_sai/splatfacto/2024-12-30_190922/dataparser_transforms.json'
    with open(splat_transforms_pth, 'r') as f:
        splat_tf = json.load(f)
    splat_Rt = np.array(splat_tf['transform']) # Gives 3x4 mat
    splat_Rt = np.vstack((splat_Rt, np.array([0,0,0,1]))) # Get 4x4 mat
    splat_scale = splat_tf['scale']
    # Invert the transformation
    splat_Rt = np.linalg.inv(splat_Rt)
    splat_scale = 1/splat_scale

    
    #! Environment 3D
    env3d = Env3D(th=dist_th , score_th=score_th)
    
    count = 0
    for file in tqdm(files):
        # count += 1
        # if count%10!=0:
        #     continue
        #! Read data
        fname = file[:-4]
        image = cv2.imread(data_dir/'rgb'/f"{fname}.jpg")
        # depth = cv2.imread(data_dir/'depth'/f"{fname}.png", cv2.IMREAD_UNCHANGED)
        depth = np.load(data_dir/'depth'/f"{fname}.npy")
        det = np.loadtxt(data_dir/'detection'/f"{fname}.txt").reshape(-1,15)
        # print(fname, det.shape)
        pose = np.loadtxt(data_dir/'pose'/f"{fname}.txt")
        seg_mask = cv2.imread(data_dir/'mask'/f"{fname}.png", cv2.IMREAD_UNCHANGED)
        # print(depth.min(), depth.max(), depth.dtype)
        # print(image.shape, depth.shape, det.shape, pose.shape)
        
   
        #! Transformations 
        # Previously
        # trans, quat = pose[:3], pose[3:]
        # rotmat = qvec2rotmat(quat)
        
        # Now 
        rotmat, trans = pose[:9].reshape(3,3), pose[9:]
        pose = np.hstack((rotmat, trans.reshape(3,1)))
        pose_4by4 = np.vstack((pose, np.array([0,0,0,1])))
        pose_inv_4by4 = np.linalg.inv(pose_4by4)
        pose_inv = pose_inv_4by4[:3]
    
        #! Detections
        uv = det[:,4:6]
        bbox = det[:,:4].astype(np.int16)
        rotmat = det[:,6:]
        no_of_obj = det.shape[0]
        
        #! Get depths
        # depth_vals, good_depth_mask = get_depth_value(bbox, depth, filter=(depth_near_th, depth_far_th))
        depth_vals, good_depth_mask, _ = get_depth_value(
                bbox, depth, seg_mask, splat_scale, depth_near_th, depth_far_th, False
        )
        
        #! Filter bad depth 
        depth_vals = depth_vals[good_depth_mask]
        uv = uv[good_depth_mask]
        rotmat = rotmat[good_depth_mask]
        
        # Ignre sample if none of the detetions are good.
        if depth_vals.shape[0]==0:
            print("ignoring sample:", fname)
            continue
       
        #! Project 2d to 3d
        points3d_cam = get_points3d(uv, depth_vals, K)
        pose_mat_cam = get_pose_mat(np.hstack((points3d_cam, rotmat)))
        pose_mat = pose_cam_to_world(pose_mat_cam, pose_4by4)
        
        trans_vec = pose_mat[:,:3,3]
        rot_mat = pose_mat[:,:3,:3]
        quat_vec = rotmat2qvec(rot_mat)
        
        env3d.add_measurement(trans_vec, quat_vec)
        
    all_xyz, all_quat = env3d.get_final_data()
   
    env3d.save_filtered_data() 
    env3d.save_measurements()
    exit()
    
    #! Plotting
    # fig, ax = plt.subplots(1,1, figsize=(10,10))
    # ax.remove()
    # ax = fig.add_subplot(111, projection='3d')
    # plot_3D_poses(ax, all_xyz, all_quat)
    # plt.title("Flower Distribution")
    # plt.show()
    
    rotmat = R.from_quat(all_quat).as_matrix()
    obj_dirn = np.array([0,0,2/100])
    rotated_obj = rotmat@obj_dirn
    dirn_xyz = all_xyz + rotated_obj
    
    xyz_homo = np.hstack((all_xyz, np.ones(all_xyz.shape[0]).reshape(-1,1)))
    dir_homo = np.hstack((dirn_xyz, np.ones(dirn_xyz.shape[0]).reshape(-1,1)))
    
    #! Colors
    colors = generate_rainbow_colors(xyz_homo.shape[0])
    colors = np.array(colors)*255.0
    np.random.shuffle(colors)

   
    #! Plot the Projection 
    for file in tqdm(files[60:]):
        # count += 1
        # if count%10!=0:
        #     continue
        #! Read data
        image = cv2.imread(dataset_dir/'rgb'/file)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        dimming_factor = 0.5
        image = (image * dimming_factor).astype(np.uint8)
        pose = np.loadtxt(dataset_dir/'pose'/f"{file[:-3]}txt")
        # print(image.shape, depth.shape, det.shape, pose.shape)
   
        #! Transformations 
        trans, quat = pose[:3], pose[3:]
        rotmat = qvec2rotmat(quat)
        pose = np.hstack((rotmat, trans.reshape(3,1)))
        pose_4by4 = np.vstack((pose, np.array([0,0,0,1])))
        pose_inv_4b4 = np.linalg.inv(pose_4by4)
        pose_inv = pose_inv_4b4[:3]
        
        #! Project Points
        xyz_proj = (K@pose_inv@xyz_homo.T).T
        dir_proj = (K@pose_inv@dir_homo.T).T
        
        xyz_proj /= xyz_proj[:,2].reshape(-1,1)
        dir_proj /= dir_proj[:,2].reshape(-1,1)
        
        xyz_proj = xyz_proj[:,:2]
        dir_proj = dir_proj[:,:2]
        
        dirn = dir_proj - xyz_proj
        
        #! Filter out points that lies outside the frame
        x = xyz_proj[:,0]
        y = xyz_proj[:,1]
        
        mask1 = np.logical_and(x<1920, x>=0)
        mask2 = np.logical_and(y<1080, y>=0)
        mask = np.logical_and(mask1, mask2)
        
        xyz_proj_masked = xyz_proj[mask]
        dir_proj_masked = dir_proj[mask]
        colors_masked = colors[mask]
        
        for st,ed,col in zip(xyz_proj_masked, dir_proj_masked, colors_masked):
            st = st.astype(np.int)
            ed = ed.astype(np.int)
            print(st, ed, col)
            cv2.arrowedLine(image, st, ed, color=(0,0,255), thickness=10, tipLength=0.2*1.3)
            cv2.arrowedLine(image, st, ed, color=col, thickness=5, tipLength=0.2)
       
        cv2.imwrite(f"/home/rashik/Downloads/walmart_purchase/flowerur_data/anno_pose/{file}", image) 
        
        # plt.clf() 
        # plt.imshow(image)
        # for st,ed,col in zip(xyz_proj_masked, dir_proj_masked, colors_masked):
        #     data = np.array([st, ed])
        #     plt.arrow(st[0], st[1], ed[0]-st[0], ed[1]-st[1], width=8, head_width=18, head_length=13, color='r')
        #     plt.arrow(st[0], st[1], ed[0]-st[0], ed[1]-st[1], width=5, head_width=15, head_length=10, color=col)
            
        # plt.axis('off')
        # plt.savefig(f"/home/rashik/Downloads/walmart_purchase/flowerur_data/anno_pose/{file}", format="png", bbox_inches='tight', pad_inches=1, dpi=300)
        # plt.show()
        
        
    