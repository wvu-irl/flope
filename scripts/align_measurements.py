"""
This script align the measurements across the measurements.
Deals with flower reidentification (based on 3d coordinates) to align the
measuresement of same flower across the frames.

Also takes average of all the aligned flower positions.
- translation is simply averaged
- for rotation, quaternion is averaged using Slerp
"""

import random
import numpy as np
import cv2
import pickle
from tqdm import tqdm
from scipy.spatial.distance import cdist
import pickle
from pathlib import Path

#! sunflower imports
from sunflower.utils.mvg import rot_average, get_points3d, pose_cam_to_world
from sunflower.utils.image_manipulation import get_depth_value
from sunflower.utils.conversion import get_pose_mat, qvec2rotmat, rotmat2qvec
from sunflower.utils.io import DatasetPath, pth, read_intrinsics_yaml_to_K_h_w


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

        if min_idx_good.shape[0] == 0: #! Warning: None of the new flower positions matched with old one!
            self.trans = np.vstack((self.trans, tvec))
            self.quat = np.vstack((self.quat, qvec))
            self.score = np.concatenate((self.score, np.ones(tvec.shape[0])))

        else:
            tvec_good = tvec[good_match]
            qvec_good = qvec[good_match]
            
            #! Append State
            state_score = self.score[min_idx_good]
            meas_score = np.ones(state_score.shape[0])
            normalizer = state_score + meas_score
            state_weight = state_score/normalizer
            meas_weight = meas_score/normalizer

            
            #! Update Trans state
            self.trans[min_idx_good] = self.trans[min_idx_good]*state_weight.reshape(-1,1) + tvec_good*meas_weight.reshape(-1,1)
            
            #! Update Quat state
            self.quat[min_idx_good] = rot_average(self.quat[min_idx_good], qvec_good, state_weight, meas_weight)
            
            #! Log the newly added measurements
            new_trans = np.zeros_like(self.trans)
            new_trans[min_idx_good] = tvec_good
            self.all_new_trans.append(new_trans)

            new_quat = np.zeros_like(self.quat)
            new_quat[:,-1] = np.ones(new_quat.shape[0])
            new_quat[min_idx_good] = qvec_good
            self.all_new_quat.append(new_quat)

            #! Update score
            self.score[min_idx_good] += 1 
            
            xyz_unmatches = tvec[np.logical_not(good_match)]
            quat_unmatches = qvec[np.logical_not(good_match)]
            self.trans = np.vstack((self.trans, xyz_unmatches))
            self.quat = np.vstack((self.quat, quat_unmatches))
            self.score = np.concatenate((self.score, np.ones(xyz_unmatches.shape[0])))


    def save_filtered_data(self, filename):
        print(f"Total unique flowers: {self.trans.shape[0]}")
        print(f"Max score: {self.score.max()}")
        print(f"Min score: {self.score.min()}")
        data = {
            'trans': self.trans,
            'quat': self.quat,
            'score': self.score
        }
        with open(filename, 'wb') as fp:
            pickle.dump(data, fp)
        print(f"Filtered data saved to {filename}")


    def get_final_data(self):
        score_filter = self.score>self.score_th
        return self.trans[score_filter], self.quat[score_filter]


    def pad_measurements(self, trans, quat):
        max_point = trans[-1].shape[0]
        print("Max points per frame: ", max_point)

        trans_paded = []
        quat_paded = []
        for tr,qa in zip(trans,quat):
            npoints = tr.shape[0]
            pad = max_point-npoints
            
            trans_pad = np.zeros((pad, 3))
            quat_pad = np.hstack((trans_pad, np.ones(pad).reshape(-1,1)))
            
            new_trans = np.vstack((tr, trans_pad))
            new_quat = np.vstack((qa, quat_pad))
            
            trans_paded.append(new_trans)
            quat_paded.append(new_quat)

        trans_paded = np.array(trans_paded)
        quat_paded = np.array(quat_paded)

        return trans_paded, quat_paded


    def save_measurements(self, filename):
        trans_paded, quat_paded = self.pad_measurements(self.all_new_trans, self.all_new_quat)
        with open(filename, 'wb') as fp:
            all_data = {
                'trans': trans_paded,
                'quat': quat_paded
            }
            pickle.dump(all_data, fp)
        print(f"All measurements saved to {filename}")
        

if __name__=='__main__':
    #! Configs
    data = DatasetPath('/home/rashik_shrestha/data/andy_data/single_cluster')
    distance_th = 50
    score_th = 100
    
    #! Random Seeds
    random.seed(0)
    np.random.seed(0)
    
    #! I/O
    data.aligned.mkdir(exist_ok=True, parents=True)
    (data.dataset/'points_3d').mkdir(exist_ok=True, parents=True)
    K,h,w = read_intrinsics_yaml_to_K_h_w(data.intrin)
    
    #! Environment 3D
    env3d = Env3D(th=distance_th , score_th=score_th)
    
    count = 0
    for file in tqdm(data.files):
        # count += 1
        # if count%10!=0:
        #     continue
        #! Read data
        image = cv2.imread(pth(data.rgb,file,'png'))
        depth_info = np.loadtxt(pth(data.depth_val,file,'txt'))
        det = np.loadtxt(pth(data.det,file,'txt'))
        pose = np.loadtxt(pth(data.pose,file,'txt'))
        # print(image.shape, depth_info.shape, det.shape, pose.shape)
        
        if det.shape[0] == 0 or depth_info.shape[0]==0:
            np.savetxt(pth(data.dataset/'points_3d', file, 'txt'), np.array([]))
            continue
        
        if len(depth_info.shape) == 1:
            depth_info = depth_info[None].T
            
        depth_val,depth_reliable = depth_info
        
        if len(det.shape) == 1:
            det = det[None]
 
        depth_reliable = depth_reliable>0.5
   
        #! Transformations 
        trans, quat = pose[:3], pose[3:] #TODO: quaternion used here, this should be rotmat
        rotmat = qvec2rotmat(quat)
        pose = np.hstack((rotmat, trans.reshape(3,1)))
        pose_4by4 = np.vstack((pose, np.array([0,0,0,1])))
        pose_inv_4b4 = np.linalg.inv(pose_4by4)
        pose_inv = pose_inv_4b4[:3]
    
        #! Detections
        uv = det[:,4:6]
        rotmat = det[:,6:]
        
        #! Filter bad depth 
        depth_val = depth_val[depth_reliable]
        uv = uv[depth_reliable]
        rotmat = rotmat[depth_reliable]

        # Ignre sample if none of the detetions are good.
        if depth_val.shape[0]==0:
            print("ignoring sample:", file)
            np.savetxt(pth(data.dataset/'points_3d', file, 'txt'), np.array([]))
            continue
      
        #! Project 2d to 3d
        points3d_cam = get_points3d(uv, depth_val, K)
        pose_mat_cam = get_pose_mat(np.hstack((points3d_cam, rotmat)))
        pose_mat = pose_cam_to_world(pose_mat_cam, pose_4by4)
        
        trans_vec = pose_mat[:,:3,3]
        rot_mat = pose_mat[:,:3,:3]
        quat_vec = rotmat2qvec(rot_mat)
        
        #! Export 3D points here
        np.savetxt(pth(data.dataset/'points_3d', file, 'txt'), trans_vec)
        
        env3d.add_measurement(trans_vec, quat_vec)
        
    all_xyz, all_quat = env3d.get_final_data()

    env3d.save_measurements(data.aligned/'measurements.pkl')
    env3d.save_filtered_data(data.aligned/'average_poses.pkl')