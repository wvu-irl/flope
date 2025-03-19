import argparse
from scipy.spatial.transform import Rotation as R
import numpy as np
import random
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True, precision=2)

from sunflower.utils.mvg import slerp_interpolate, procustus_interpolate

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Rotation Interpolation")
    parser.add_argument('--seed', type=int, default=None, help="Random seed (integer value).")
    parser.add_argument('--r1', type=float, nargs=3, help="Rotation 1 (e.g., --r1 15.0 90.0 30.5).")
    parser.add_argument('--r2', type=float, nargs=3, help="Rotation 2 (e.g., --r1 15.0 90.0 30.5).")
    parser.add_argument('--out', type=str, default='rot_interp.png' , help="Output Image Path")
    args = parser.parse_args()

    #! Select rotation1 and rotation2 based on given input
    if args.seed != None:
        print(f"Using Random seed: {args.seed}")
        np.random.seed(args.seed)
        random.seed(args.seed)
        rot1 = R.random()
        rot2 = R.random()
    elif args.r1 != None and args.r2 != None:
        rot1 = R.from_euler('xyz', np.array(args.r1), degrees=True)
        rot2 = R.from_euler('xyz', np.array(args.r2), degrees=True)
    else:
        rot1 = R.from_euler('xyz', [0, 0, 0], degrees=True)
        rot2 = R.from_euler('xyz', [120, 0, 0], degrees=True)

    print("Rot1:", rot1.as_euler('xyz'))
    print("Rot2:", rot2.as_euler('xyz'))

    #! Indices
    num_interpolations = 100
    indices = np.linspace(0, 1, num_interpolations)

    #! Do Interpolation
    all_mat, all_angles = slerp_interpolate(rot1, rot2, indices)
    interp_matrix, interp_rotmat, interp_angles = procustus_interpolate(rot1, rot2, indices)

    #! Plot
    fig, axs = plt.subplots(3,3,figsize=(10,10))

    for i in range(3):
        for j in range(3):
            ax = axs[i,j]
            ax.plot(all_mat[:,i,j], label='slerp')
            ax.plot(interp_matrix[:,i,j], label='linear', linestyle='dashed')
            ax.plot(interp_rotmat[:,i,j], label='projected')
            ax.grid()
            if i==0 and j==0:
                ax.legend()
                
    # Title
    begin_euler = rot1.as_euler('xyz', degrees=True)
    end_euler = rot2.as_euler('xyz', degrees=True)
    title = f"Eulers(xyz): {begin_euler}->{end_euler}"
    if args.seed:
        title = f"Seed: {args.seed} | " + title
    fig.suptitle(title)

    fig.tight_layout()
    plt.savefig(args.out)
    print(f"Plot saved to: {args.out}")