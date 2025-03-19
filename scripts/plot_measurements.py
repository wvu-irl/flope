import numpy as np
import cv2
import pickle
from scipy.spatial.transform import Rotation as scipyR
from tqdm import tqdm
from tyro import cli

from sunflower.utils.io import DatasetPath, pth, read_intrinsics_yaml_to_K_h_w, render_video
from sunflower.utils.mvg import make_homogeneous
from sunflower.utils.conversion import qvec2rotmat
from sunflower.utils.plot import generate_rainbow_colors


def main(
    data: str = '/home/rashik_shrestha/data/sunflower/flowerur',
    meas: str = 'measurements'
):
    data = DatasetPath(data)
    print(f"{len(data.files)} frames available.")
    
    #! Read Intrinsics
    K,h,w = read_intrinsics_yaml_to_K_h_w(data.intrin)
   
    #! Read flower poses
    with open(pth(data.aligned, meas, 'pkl'), 'rb') as fp:
        meas_data = pickle.load(fp)
    flower_trans, flower_quat = meas_data['trans'], meas_data['quat']
    
    #! Get colors
    colors = generate_rainbow_colors(flower_trans.shape[1])*255
    ids = np.arange(flower_trans.shape[1])
    
    #! Output dir
    out_dir = data.aligned/meas
    out_dir.mkdir(exist_ok=True, parents=True)

    
    #! Flower direction
    flower_dirn_x = np.array([2/100,0,0])
    flower_dirn_y = np.array([0,2/100,0])
    flower_dirn = np.array([0,0,2/100])
    
    for i in tqdm(range(len(data.files))):
        #! Read data
        file = data.files[i]
        image = cv2.imread(pth(data.rgb, file, 'png'))
        cam_pose = np.loadtxt(pth(data.pose, file, 'txt'))
        ftrans, fquat = flower_trans[i], flower_quat[i]
        # print(image.shape, cam_pose.shape, ftrans.shape, fquat.shape)
        
        #! For Flower Poses
        # print(ftrans.shape, fquat.shape)
        mask = np.sum(np.abs(ftrans), axis=-1) != 0
        ftrans = ftrans[mask]
        fquat = fquat[mask]
        fcol = colors[mask] 
        fid = ids[mask]
        
        frotmat = scipyR.from_quat(fquat).as_matrix()
        # print(ftrans.shape, fquat.shape, frotmat.shape)
        fdir_trans_x = frotmat@flower_dirn_x + ftrans
        fdir_trans_y = frotmat@flower_dirn_y + ftrans
        fdir_trans = frotmat@flower_dirn + ftrans
        # print(fdir_trans.shape)
        ftrans_homo = make_homogeneous(ftrans)
        fdir_trans_x_homo = make_homogeneous(fdir_trans_x)
        fdir_trans_y_homo = make_homogeneous(fdir_trans_y)
        fdir_trans_homo = make_homogeneous(fdir_trans)
        # print(ftrans_homo.shape, fdir_trans_homo.shape)
        
        #! For Camera Poses
        ctrans, cquat = cam_pose[:3], cam_pose[3:]
        crotmat = qvec2rotmat(cquat)
        cpose = np.hstack((crotmat, ctrans.reshape(3,1)))
        cpose_4by4 = np.vstack((cpose, np.array([0,0,0,1])))
        cpose_inv_4b4 = np.linalg.inv(cpose_4by4)
        cpose_inv = cpose_inv_4b4[:3]
        
        #! Project Points
        ftrans_proj = (K@cpose_inv@ftrans_homo.T).T
        ftrans_dir_x_proj = (K@cpose_inv@fdir_trans_x_homo.T).T
        ftrans_dir_y_proj = (K@cpose_inv@fdir_trans_y_homo.T).T
        ftrans_dir_proj = (K@cpose_inv@fdir_trans_homo.T).T
        
        ftrans_proj /= ftrans_proj[:,2].reshape(-1,1)
        ftrans_dir_x_proj /= ftrans_dir_x_proj[:,2].reshape(-1,1)
        ftrans_dir_y_proj /= ftrans_dir_y_proj[:,2].reshape(-1,1)
        ftrans_dir_proj /= ftrans_dir_proj[:,2].reshape(-1,1)
        
        ftrans_proj = ftrans_proj[:,:2]
        ftrans_dir_x_proj = ftrans_dir_x_proj[:,:2]
        ftrans_dir_y_proj = ftrans_dir_y_proj[:,:2]
        ftrans_dir_proj = ftrans_dir_proj[:,:2]
        # print(ftrans_proj.shape, ftrans_dir_proj.shape)
       
        #! Plot direction points to image
        for id,st,edx,edy,ed,col in zip(fid, ftrans_proj, ftrans_dir_x_proj, ftrans_dir_y_proj, ftrans_dir_proj, fcol):
            st = st.astype(np.int16)
            edx = edx.astype(np.int16)
            edy = edy.astype(np.int16)
            ed = ed.astype(np.int16)
            # print(st, ed, col)
            cv2.line(image, st, edx, color=(0,0,255), thickness=10)
            cv2.line(image, st, edy, color=(0,255,0), thickness=10)
            cv2.line(image, st, ed, color=(255,0,0), thickness=10)
            cv2.putText(image, str(id), st, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

            # cv2.arrowedLine(image, st, ed, color=(0,0,255), thickness=10, tipLength=0.2*1.3)
            # cv2.arrowedLine(image, st, ed, color=col, thickness=5, tipLength=0.2)
            
        cv2.imwrite(pth(out_dir, file, 'png'), image)
        
    render_video(out_dir) 
    
if __name__=='__main__':
    cli(main)