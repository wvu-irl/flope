import numpy as np
from tyro import cli
import pickle
import matplotlib.pyplot as plt

from sunflower.utils.io import DatasetPath, pth

def main(
    path: str = '/home/rashik_shrestha/data/sunflower/flowerur'
):
    data = DatasetPath(path)
    with open(data.aligned/'measurements.pkl', 'rb') as fp:
        meas_raw = pickle.load(fp)
        
    with open(data.aligned/'measurements_quat_filter.pkl', 'rb') as fp:
        meas_filter = pickle.load(fp)
       
    idx = 0
        
    trans_raw = meas_raw['trans'][:,idx,:]
    mask = np.sum(np.abs(trans_raw), axis=-1) != 0
    quat_raw = meas_raw['quat'][:,idx,:]
    
    data_raw = np.hstack((trans_raw, quat_raw))
    
    trans_filter = meas_filter['trans'][:,idx,:]
    quat_filter = meas_filter['quat'][:,idx,:]
    data_filter = np.hstack((trans_filter, quat_filter))
    
    data_raw = data_raw[mask]
    data_filter = data_filter[mask]

    labels = ['X', 'Y', 'Z', 'q0', 'q1', 'q2', 'q3']
    
    fig, axs = plt.subplots(7,1, figsize=(10,15))
    
    print('Plotting...')
    for i in range(7):
        ax = axs[i]
        ax.plot(data_raw[:,i], linestyle=":")
        ax.plot(data_filter[:,i])
        ax.set_ylabel(labels[i])
    
    plt.savefig('raw_vs_filtered.png')
    print('done')
    
if __name__=='__main__':
    cli(main)