import numpy as np
from scipy.spatial.transform import Rotation as sciR
import roma
import torch

#! qvec2rotmat_colmap and rotmat2qvec_colmap differs for scipy and roma ones!!
# This differs from quaternion to rotation matrix conversion as provided
# by scipy and roma. TODO: Find what's the difference?
# Try to make less use of these functions.
def qvec2rotmat_colmap(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def rotmat2qvec_colmap(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def qvec2rotmat(quat):
    return sciR.from_quat(quat).as_matrix()


def rotmat2qvec(rotmat):
    return sciR.from_matrix(rotmat).as_quat()


def R2E(R):
    """Rotation matrix (3x3) to Euler angles (3,)"""
    return sciR.from_matrix(R).as_euler('zyx', degrees=True)  

def E2R(E):
    """Euler angles (3,) to Rotation matrix (3x3)"""
    return sciR.from_euler('zyx', E, degrees=True).as_matrix()


def procrustes_to_rotmat(inp: torch.Tensor) -> torch.Tensor:
    """
    Works with a batch of procustes as well.
    """
    return roma.special_procrustes(inp.reshape(-1, 3, 3))


def get_pose_mat(trans_rot):
    """
    Args:
        trans_rot: (N,12) 3 trans and 9 rot
    Returns
        pose: (N,4,4) Pose Matrix
    """
    pose_matrices = []
    for tr in trans_rot:
        tvec = tr[:3].reshape(3,1)
        rotmat = tr[3:].reshape(3,3)
        pose = np.hstack((rotmat, tvec))
        pose = np.vstack((pose, np.array([0,0,0,1])))
        pose_matrices.append(pose)
    pose_matrices = np.array(pose_matrices)
    return pose_matrices


def openCV_to_openGL_c2w(pose):
    """
    Parameters
    ----------
    pose: np.ndarray
        (4,4) c2w i.e Pose of the camera
    """
    pose[0:3, 1:3] *= -1
    pose = pose[np.array([1, 0, 2, 3]), :]
    pose[2, :] *= -1
    return pose


def openGL_to_openCV_c2w(pose):
    """
    Parameters
    ----------
    pose: np.ndarray
        (4,4) c2w i.e Pose of the camera
    """
    pose[2, :] *= -1
    pose = pose[np.array([1, 0, 2, 3]), :]
    pose[0:3, 1:3] *= -1
    return pose