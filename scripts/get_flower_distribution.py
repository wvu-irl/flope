import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

np.set_printoptions(suppress=True, precision=5)

#! Parameters
plot = False

#! I/O
with open('meas.pkl', 'rb') as fp:
    data = pickle.load(fp)
trans = data['trans']    
quat = data['quat']

#! some quaternions are [0,0,0,0], replace it by [0,0,0,1]
for q1 in quat:
    for q2 in q1:
        if q2[0]==0 and q2[1]==0 and q2[2]==0 and q2[3]==0:
            q2[3]=1

max_point = trans[-1].shape[0]
print("Max points per frame: ", max_point)

#! Add paddings
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


#! Quat to Rot Mat
F, N, qd = quat_paded.shape
rot_paded = R.from_quat(quat_paded.reshape(F*N,4)).as_matrix().reshape(F,N,3,3).reshape(F,N,9)
all_paded = np.concatenate((trans_paded, rot_paded), axis=2)


#! Calculate mean variances of pose distribution in each of 12 dimensions and plot distribution
if plot:
    labels = ['X', 'Y', 'Z', 'R00', 'R01', 'R02', 'R10', 'R11', 'R12', 'R20', 'R21', 'R22']

all_var = []
for i in tqdm(range(40)): # for each point
    if plot:
        fig, ax = plt.subplots(4,3, figsize=(9,12))
    point = all_paded[:, i, :]
    np.savetxt('points.txt', point)
   
    this_var = [] 
    done = False
    for j in range(12): # for each dimension
        data = point[:,j]
        if j==3 or j==7 or j== 11:
            data = data[data!=1]
        else:
            data = data[data!=0]
        
        if len(data) == 0:
            break
    
        this_var.append(data.var())
        
        done = True

        if plot:
            ax[j//3,j%3].hist(data)
            ax[j//3,j%3].set_title(labels[j])
    
    if done: 
        all_var.append(this_var)

    if plot:
        plt.subplots_adjust(hspace=0.35)
        plt.suptitle(f"Distribution of a 3D point {i} across {F} frames.")
        plt.tight_layout()
        plt.savefig(f"points_dist/{i:2d}.png")
    
all_var = np.array(all_var)
all_var_mean = np.mean(all_var, axis=0)

#! Save the calculated variances
np.savetxt('variances.txt', all_var_mean)
print(all_var_mean)