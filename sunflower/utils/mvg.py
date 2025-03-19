import numpy as np
import torch
import cv2
from scipy.spatial.transform import Rotation as R, Slerp

from jaxtyping import Float, jaxtyped
from beartype import beartype

from sunflower.utils.conversion import qvec2rotmat, rotmat2qvec, R2E, E2R, procrustes_to_rotmat

@jaxtyped(typechecker=beartype)
def make_homogeneous(points: Float[np.ndarray, "N ..."]) -> Float[np.ndarray, "N ..."]:
    """Adds a ones column at the end
    
    Args:
        points: N points. Can be (N,2) or (N,3)
        
    Returns:
        Homogeneous points. Can be (N,3) or (N,4)
    """
    N = points.shape[0]
    ones_column = np.ones(N).reshape(-1,1)
    homogeneous = np.hstack((points, ones_column))
    return homogeneous


@jaxtyped(typechecker=beartype)
def pixel_to_camera_coord(
    uv: Float[np.ndarray, "N 2"], 
    d: Float[np.ndarray, "N"], 
    K: Float[np.ndarray, "3 3"]
) -> Float[np.ndarray, "N 3"]:
    """Pixel to Cam coordinate
    
    Args:
        uv: Pixel coordinates
        d: Depth values (preferrd unit in meters)
        K: Camera Intrinsic Matrix
    
    Returns:
        Camera Coordinates (same unit as depth d)
    """
    uv = make_homogeneous(uv)
    uv *= d.reshape(-1,1)
    xyz = (np.linalg.inv(K)@uv.T).T
    return xyz


@jaxtyped(typechecker=beartype)
def camera_to_world_coord(
    xyz: Float[np.ndarray, "N 3"],
    Rctc: Float[np.ndarray, "4 4"]
) -> Float[np.ndarray, "N 3"]:
    """Camera to world coord
    
    Args:
        xyz: Camera coordinates
        Rctc: Camera Pose
    
    Returns:
        World coordinates
    """
    xyz = make_homogeneous(xyz)
    xyz_world = (Rctc@xyz.T).T
    return xyz_world[:,:-1]


def project_3d_to_2d(points, K, R, t):
    """
    Project set of 3d points to 2d image
    """
    t = t.reshape(-1,1)
    points_trans = R@points.T + t
    projection = K@points_trans
    projection /= projection[2]
    projection = projection[:2]
    return projection.T


def get_bounding_box_from_reprojected_box(corners):
    """
    From a projected cube to a image, get bounding box.
    The bounding box will be square.
    """
    xmin = np.min(corners[:,0])
    xmax = np.max(corners[:,0])
    ymin = np.min(corners[:,1])
    ymax = np.max(corners[:,1])

    xrange = xmax-xmin
    yrange = ymax-ymin
    diff = abs(xrange-yrange)

    if xrange>yrange:
        ymin -= diff/2
        ymax += diff/2
    else:
        xmin -= diff/2
        xmax += diff/2
    
    # Do the same in integer
    xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
    xrange = xmax-xmin
    yrange = ymax-ymin
    diff = abs(xrange-yrange)

    if xrange>yrange:
        ymax += diff
    else:
        xmax += diff

    return [xmin, ymin, xmax, ymax]


def transform_corners(h,w,H):
    # Define the 4 corner points of the original image
    original_corners = np.array([
        [0, 0],         # Top-left corner
        [w - 1, 0],     # Top-right corner
        [w - 1, h - 1], # Bottom-right corner
        [0, h - 1]      # Bottom-left corner
    ], dtype=np.float32)

    # Reshape the corner points to the required shape for perspectiveTransform
    original_corners = original_corners.reshape(-1, 1, 2)

    # Apply the homography transformation
    transformed_corners = cv2.perspectiveTransform(original_corners, H)

    # Reshape the transformed corners back to (N, 2) for easier interpretation
    transformed_corners = transformed_corners.reshape(-1, 2)
    
    return transformed_corners
    

def rotate_image(image, rotation_angles, camera_intrinsics):
    """
    Rotate the input image by specified angles using homography.

    Args:
        image: The initial camera image (numpy array).
        rotation_angles: Tuple (theta_x, theta_y, theta_z) for rotation in degrees.
        camera_intrinsics: The intrinsic matrix of the camera.

    Returns:
        np.ndarray: Rotated Image (3,H,W) or (H,W)
        np.ndarray: Rotation Matrix (3,3)
    """
    # Convert rotation angles to radians
    theta_x, theta_y, theta_z = np.radians(rotation_angles)

    # Define rotation matrices around X, Y, and Z axes
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta_x), -np.sin(theta_x)],
                    [0, np.sin(theta_x), np.cos(theta_x)]])
    
    R_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                    [0, 1, 0],
                    [-np.sin(theta_y), 0, np.cos(theta_y)]])
    
    R_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                    [np.sin(theta_z), np.cos(theta_z), 0],
                    [0, 0, 1]])

    # Combine rotations
    R = R_z @ R_y @ R_x

    # Compute homography matrix
    H = camera_intrinsics @ R @ np.linalg.inv(camera_intrinsics)

    # Warp the image using the homography matrix
    h, w = image.shape[:2]
    rotated_image = cv2.warpPerspective(image, H, (w, h))

    # Transform the four corners of image under H    
    transformed_corners = transform_corners(h,w,H)
    
    return rotated_image, R, transformed_corners, H


def homography_transform(image, rotation_angles, intrinsics, mask=None, Rs=None, ts=None):
    """
    Apply Homography Transformation
    
    Args:
        image (np.ndarray): Image (3,H,W)
        rotation_angles (np.ndarray): Rotation euler angles (Rx, Ry, Rx)
        intrinsics (np.ndarray): Camera Intrinsincs (3,3)
        mask (np.ndarray): Image mask (H,W)
        poses (np.ndarray): Object poses in the form quaternion and trans (N,7)
        
    Returns:
        np.ndarray: Transformed Image (3,H,W)
        np.ndarray: Transformed Mask (H,W)
        np.ndarray: Object poses in the form quaternion and trans (N,7)
    """
    rotated_image, R_rot, transform_corners, H = rotate_image(image, rotation_angles, intrinsics)
    rotated_mask, rotated_poses = None, None
    if mask is not None:
        rotated_mask, _, _,H = rotate_image(mask, rotation_angles, intrinsics)
    if Rs is not None and ts is not None:
        Rs_new, ts_new = [], []
        for R,t in zip(Rs,ts):
            Rnew = R_rot@R
            tnew = (R_rot@t.reshape(-1,1)).squeeze()
            Rs_new.append(Rnew)
            ts_new.append(tnew)
        
    return rotated_image, rotated_mask, np.array(Rs_new), np.array(ts_new), transform_corners


    # if poses is not None:
    #     all_q, all_t = poses[:,:4], poses[:,4:]
    #     all_q_new, all_t_new = [], []
    #     for q,t in zip(all_q,all_t):
    #         Rmat = qvec2rotmat(q)
    #         Rnew = R@Rmat
    #         tnew = (R@t.reshape(-1,1)).squeeze()
    #         qnew = rotmat2qvec(Rnew)
    #         all_q_new.append(qnew)
    #         all_t_new.append(tnew)
    #     all_q_new = np.array(all_q_new) 
    #     all_t_new = np.array(all_t_new) 
    #     rotated_poses = np.hstack((all_q_new, all_t_new))


def nullify_yaw(Rmatrix):
    """
    Args:
        Rmatrix (np.ndarray): Rotation Matrix (3x3)
    Returns:
        np.ndarray: Yaw nullified rotation matrix (3x3)
    """
    euler_angles = R2E(Rmatrix)
    euler_angles[0] = 0.0 # Nullify Yaw rotation
    Rmatrix_yaw_nullified = E2R(euler_angles)
    return Rmatrix_yaw_nullified


@jaxtyped(typechecker=beartype)
def nullify_yaw_batch(rotmat: Float[np.ndarray, "N 3 3"])-> Float[np.ndarray, "N 3 3"]:
    """
    Args:
        rotmat: rotation matrices
    Returns:
        Yaw nullified rotation matrices
    """
    euler_angles = R2E(rotmat)
    euler_angles[:,0] = 0.0 # Nullify Yaw rotation
    rotmat_yaw_nullified = E2R(euler_angles)
    return rotmat_yaw_nullified


def get_crop_intrinsics(K, offset, scale):
    """
    Args:
        offset (tuple): (xmin, ymin) values of the crop, you dont need (xmax, ymax)
    Returns:
        np.ndarray: Camera Intrinsic Matrix of cropped image, (3x3)
    """
    crop_intrin = np.copy(K)
    crop_intrin[0,2] -= offset[0]
    crop_intrin[1,2] -= offset[1]
    crop_intrin = scale*crop_intrin
    crop_intrin[2,2] = 1.0 #! Dont bloody hell forget this while scaling!!!!!!!!!!!
    return crop_intrin
    
    
def slerp_interpolate(r1, r2, indices):
    """
    Args:
        r1: Start rotation
        r2: End rotation
        indices: N Interpolation indices, range [0,1]
    
    Returns:
        all_mat: N interpolated rotation matrix
        all_angles: Angles between r1 and all_mat
    """
    times = [0, 1]
    slerp = Slerp(times, R.concatenate([r1, r2]))

    all_mat = []
    all_angles = []
    for i in indices:
        rot3 = slerp([i])[0]
        all_mat.append(rot3.as_matrix()) 
        R_relative = r1.inv() * rot3
        angle = R_relative.magnitude()
        angle_degrees = np.degrees(angle)
        all_angles.append(angle_degrees)
        
    all_angles = np.array(all_angles)
    all_mat = np.array(all_mat) 
    
    return all_mat, all_angles


def procustus_interpolate(r1, r2, indices):
    rot1_r9 = r1.as_matrix().flatten()
    rot2_r9 = r2.as_matrix().flatten()
   
    #! Generate Interpolated Vectors and Matrices
    interpolated_vec = np.array([
        (1 - t) * rot1_r9 + t * rot2_r9 for t in indices
    ])
    interpolated_matrix = interpolated_vec.reshape(-1, 3, 3)
   
    #! Project Interpolated Vectors to Rotation Matrices 
    interpolated_rot = procrustes_to_rotmat(torch.as_tensor(interpolated_vec)).numpy()
   
    #! Get angular difference between each inperpolated rot w.r.t r1
    all_angles = []
    for rot3 in interpolated_rot:
        rot3 = R.from_matrix(rot3)
        R_relative = r1.inv() * rot3
        angle = R_relative.magnitude()
        angle_degrees = np.degrees(angle)
        all_angles.append(angle_degrees)
    all_angles = np.array(all_angles)
    
    return interpolated_matrix, interpolated_rot, all_angles
    
def squarify_bb(bb):
    xmin, ymin, xmax, ymax = bb
    xrange = xmax-xmin
    yrange = ymax-ymin
    diff = abs(xrange-yrange)
    if diff%2 == 0:
        decrease_min = diff/2
        increase_max = diff/2
    else:
        decrease_min = (diff+1)/2 # 
        increase_max = (diff-1)/2
    if xrange>yrange:
        ymin -= decrease_min
        ymax += increase_max
    elif xrange<yrange:
        xmin -= decrease_min
        xmax += increase_max
    
    final_bb = [int(xmin), int(ymin), int(xmax), int(ymax)]
    return final_bb

def bb_in_frame(bb, img_shape):
    h,w,_ = img_shape
    xmin, ymin, xmax, ymax = bb
    if xmin<0 or ymin<0 or xmax>w or ymax>h:
        return False
    else:
        return True 
    

def filter_very_large_bb(bb_dino):
    bb_dino = np.array(bb_dino) 
    x_range = bb_dino[:,2]-bb_dino[:,0]
    y_range = bb_dino[:,3]-bb_dino[:,1]
    area = x_range*y_range
    median_area = np.median(area)
    large_area = area>5*median_area
    bb_dino = bb_dino[np.logical_not(large_area)]
    return bb_dino


def rot_average(quat1, quat2, weight1, weight2):
    """
    Args:
        quat1: (N1,4)
        quat2: (N2,4)
    """
    # print(quat1.shape, quat2.shape, weight1.shape, weight2.shape)
    N = quat1.shape[0]
   
    avg_quat = [] 
    for i in range(N):
        q1, q2, w1, w2 = quat1[i], quat2[i], weight1[i], weight2[i]
        # print(q1, q2, w1, w2)
        times = [0, 1]  # Assume q1 is at time 0 and q2 is at time 1
        slerp = Slerp(times, R.concatenate([R.from_quat(q1), R.from_quat(q2)]))
        t = w2 / (w1 + w2)
        avg_rotation = slerp([t])
        avg_quat.append(avg_rotation.as_quat()[0])
    avg_quat = np.array(avg_quat)
    return avg_quat


def get_points3d(uv, Zray, K):
    """
    Args:
        uv (np.array): (N,2) Pixel cordinates
        Zray (np.array): (N,) Depth values in meters
        K : (3,3) Camera Intrinsics
    Returns:
        points3d (np.array): (N,3) 3D points in camera coordinate system
    """
    N = uv.shape[0]
    uv1 = np.hstack((uv, np.ones(N).reshape(-1,1)))
    xnyn1 = (np.linalg.inv(K)@uv1.T).T
    xnyn1_norm = np.linalg.norm(xnyn1, axis=1)
    Z = Zray/xnyn1_norm
    xyz = xnyn1*Z.reshape(-1,1)
    # project back to see if everything is fine
    # uv_reproj = (K@xyz.T).T
    # uv_reproj /= uv_reproj[:,2].reshape(-1,1)
    # uv_reproj = uv_reproj[:,:2]
    # print(uv[:2])
    # print(uv_reproj[:2])
    return xyz
    
    # points2d = np.hstack((points2d, np.ones(points2d.shape[0]).reshape(-1,1)))
    # points2d *= depths.reshape(-1,1)
    # points3d = (np.linalg.inv(K)@points2d.T).T
    # return points3d


def pose_cam_to_world(obj_pose, cam_pose):
    """
    Args:
        obj_pose: (N,4,4) Object Pose in Camera coordinate system
        cam_pose: (4,4) Camera Pose matrix, or inverse of camera extrinsic matrix
    """
    return cam_pose@obj_pose