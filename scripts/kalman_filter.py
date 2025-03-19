
from pathlib import Path
import os
import pickle
from tyro import cli

from sunflower.utils.io import DatasetPath, pth

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from filterpy.kalman import KalmanFilter
from tqdm import tqdm


np.set_printoptions(suppress=True, precision=5)


def vis_flower_detecton_across_frames(meas_trans):
    """
    Args:
        meas_trans: Translation of flowers across frames. (FxNx3)
    """

    meas_trans = np.abs(meas_trans)
    meas_trans = np.sum(meas_trans, axis=-1)
    meas_trans = meas_trans != 0
    meas_trans = meas_trans.T
    nflowers, nframes = meas_trans.shape

    plt.figure(figsize=(10, 8))  # Adjust the size of the heatmap
    sns.heatmap(meas_trans, cmap='viridis', cbar=False)  # Choose a colormap, and disable the color bar
    plt.title("Flower Detection Heatmap")
    plt.xlabel("Image Frames")
    plt.ylabel("Flower IDs")
    plt.show()


def get_pose_variances(all_trans, all_quats):
    all_trans = np.transpose(all_trans, (1,0,2))
    all_quats = np.transpose(all_quats, (1,0,2))

    all_var = []

    #! For each unique flower:
    for trans, quats in zip(all_trans, all_quats):
        # Get no poses mask
        trans_sum = np.sum(np.abs(trans), axis=-1)
        mask = trans_sum != 0

        # consider this flower for variance calculation
        # if detected more than 10 times
        if mask.sum()<10:
            continue

        trans, quats = trans[mask], quats[mask]
        trans_var = np.var(trans, axis=0)
        quats_var = np.var(quats, axis=0)

        trans_quats_var = np.hstack((trans_var, quats_var))

        all_var.append(trans_quats_var)

    all_var = np.array(all_var)
    avg_var = np.mean(all_var, axis=0)
    
    return avg_var

from tqdm import tqdm
import pickle
import numpy as np
from filterpy.kalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt


def get_kalman_filter(initial_value):
    kf = KalmanFilter(dim_x=7, dim_z=7)
    kf.x = np.array(initial_value)  # Initial state estimate
    kf.F = np.eye(7)
    kf.H = np.eye(7)
    kf.P = np.eye(7)
    kf.Q = np.eye(7)*0.001
    kf.R = np.eye(7)*0.1 
    return kf


def main(
    path: str = '/home/rashik_shrestha/data/sunflower/flowerur'
):
    data = DatasetPath(path)
    
    with open(data.aligned/'measurements.pkl', 'rb') as fp:
        meas_data = pickle.load(fp)
        
    trans = meas_data['trans']    
    quat = meas_data['quat']

    F,N,_ = trans.shape
    
    kfs = [None]*N
    
    ftrans = np.zeros_like(trans)
    fquat = np.zeros_like(quat)

    print(f"{N} unique flowers.")
    
    #! For each frame
    for i in tqdm(range(F)):
        tra, qua = trans[i], quat[i]
        
        #! For each flower
        for j in range(N):
            t,q = tra[j], qua[j]
            new_meas = np.hstack((t,q))
            
            if np.sum(np.abs(t)) == 0:
                continue
           
            #! Initialize filter if not
            if kfs[j]==None: # initialize kalman filter
                kfs[j] = get_kalman_filter(new_meas)
            #! Update the current state
            else:
                kfs[j].predict()
                kfs[j].update(new_meas)
               
            #! Normalize quaternion 
            kfs[j].x[3:] /= np.linalg.norm(kfs[j].x[3:])
               
            #! Store the output 
            # ftrans[i,j] = kfs[j].x[:3]
            ftrans[i,j] = t
            fquat[i,j] = kfs[j].x[3:]
            
            
    out_file = data.aligned/'measurements_quat_filter.pkl' 
    with open(out_file, 'wb') as fp:
        all_data = {
            'trans': ftrans,
            'quat': fquat
        }
        pickle.dump(all_data, fp)
    print(f"All measurements happily saved to: {out_file} :)")
    
 
if __name__=='__main__':
    cli(main)